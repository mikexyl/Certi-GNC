/* ----------------------------------------------------------------------------
 * This file is a derivative of gtsam::GncOptimizer from GTSAM.
 *
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 *
 * Original source:
 *   gtsam/nonlinear/GncOptimizer.h
 *   Authors: Jingnan Shi, Luca Carlone, Frank Dellaert
 *
 * Modifications copyright 2026, Northeastern University Robust Autonomy Lab.
 * Distributed under the same license as the original.
 * -------------------------------------------------------------------------- */

/**
 * @file    CertifiableGncOptimizer.h
 * @brief   GNC wrapper that swaps the inner solver for the
 *          Riemannian-staircase certifiable optimizer. (Only TLS loss type for now)
 *
 * @author  Zhexin Xu
 *
 * Primary reference:
 *   Xu, Zhang, Calatrava, Closas, Rosen.
 *   "Implementing Robust M-Estimators with Certifiable Factor Graph
 *   Optimization." arXiv:2603.20932, 2026.
 *
 * Certifiable framework:
 *   Xu, Sanderson, Zhang, Rosen. "Certifiable Estimation with Factor Graphs."
 *   (Certi-fgo) arXiv:2603.01267, 2026.
 *
 * GNC algorithm:
 *   Yang, Antonante, Tzoumas, Carlone. "Graduated Non-Convexity for Robust
 *   Spatial Perception." ICRA/RAL, 2020.
 *
 * Derived from gtsam::GncOptimizer (Shi, Carlone, Dellaert). Changes:
 *   - TLS only; Geman-McClure and the non-certifiable path are removed.
 *   - Inner solver is CertifiableProblem::SolveWeighted (staircase).
 *   - Cached unweighted graph is refreshed when the certifier climbs.
 *   - Writes a per-run summary CSV with staircase diagnostics.
 */

#ifndef STIEFELMANIFOLDEXAMPLE_CERTIFIABLEGNCOPTIMIZER_H
#define STIEFELMANIFOLDEXAMPLE_CERTIFIABLEGNCOPTIMIZER_H

#include "CertifiableGncParams.h"
#include "../CertifiablePGO.h"
#include "../CertifiableLandmark.h"

#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <algorithm>
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/pointer_cast.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <set>

namespace gtsam {

/// chi^2 inverse CDF: same semantics as MATLAB chi2inv.
inline double Certifiable_Chi2inv(const double alpha, const size_t dofs) {
    return boost::math::quantile(boost::math::chi_squared(static_cast<double>(dofs)), alpha);
}

template <class CertifiableGncParamsT>
class CertifiableGncOptimizer {
public:
    typedef typename CertifiableGncParamsT::OptimizerType BaseOptimizer;

    /// Per-outer-iteration record. One row per GNC outer iteration.
    struct IterInfo {
        size_t iter_ = 0;
        double out_iter_time_ = 0.0;
        double solve_time_ = 0.0;
        double eta_ = 0.0;
        VerificationStatus verification_status_ = VerificationStatus::NOT_VERIFIED;
        double objective_ = std::numeric_limits<double>::quiet_NaN();
        Vector IterInfo_Weights_;
        CertificateResults certifiableResults_;
    };

    /// Aggregate result: final weights/estimates plus the per-iteration log.
    struct CertifiableGncResult {
        double finalTotalTime_ = 0.0;
        Vector finalWeights_;
        Values finalEstimates_;
        std::vector<IterInfo> iter_log_;
    };

private:
    NonlinearFactorGraph nfg_;        ///< Unweighted lifted graph, kept in sync with the current rank.
    Values state_;                    ///< Current best estimate (lifted Values).
    CertifiableGncParamsT params_;
    Vector weights_;                  ///< Current per-factor weights (size == nfg_.size()).
    Vector barcSq_;                   ///< Per-factor inlier-residual thresholds.
    size_t d_;                        ///< Ambient dimension (2 or 3).
    size_t currentRank_;              ///< Rank used to build the cached nfg_.
    std::shared_ptr<void> PGO_holder_;///< Type-erased shared ownership of the PGO object.

    // Function-pointer style adapters into the underlying problem. These let us
    // template the wrapper over d without rewriting the body twice.
    std::function<NonlinearFactorGraph()>                          fn_getCurrentGraph_;
    std::function<const Values&()>                                 fn_getCurrentValues_;
    std::function<void(const Values&)>                             fn_setCurrentValues_;
    std::function<FastVector<uint64_t>()>                          fn_getOdomIds_;
    std::function<size_t()>                                        fn_currentRank_;
    std::function<NonlinearFactorGraph(size_t)>                    fn_buildGraphAt_;
    std::function<std::optional<CertificateResults>(const Vector&,
                                                    size_t,
                                                    size_t,
                                                    size_t)>       fn_solveWeighted_;
    std::function<void(const std::string&)>                        fn_export_;

    std::string outputFile_;
    CertifiableGncResult result_;

public:
    /**
     * @brief Construct a GNC wrapper from any problem type that exposes the
     *        certifiable-with-weights interface (CertifiablePGO<d> or
     *        CertifiableLandmark<d>).
     *
     * ProblemT must expose: getCurrentGraph(), getCurrentValues(),
     * setCurrentValues(Values), getOdometryPoseEdgeIds(), currentRank(),
     * buildGraphAtLevel(size_t), SolveWeighted(Vector, size_t, size_t, size_t),
     * getCurrentWeights(), setCurrentWeights(Vector), clearCurrentWeights(),
     * RoundSolutionS(), ExportData(string, Matrix, bool).
     */
    template <typename ProblemT>
    CertifiableGncOptimizer(std::shared_ptr<ProblemT> problem,
                            CertifiableGncParamsT params = CertifiableGncParamsT())
        : params_(params), d_(problem->getD()), currentRank_(0), PGO_holder_(problem) {
        auto raw = problem.get();
        fn_getCurrentGraph_ = [raw]() -> NonlinearFactorGraph { return raw->getCurrentGraph(); };
        fn_getCurrentValues_ = [raw]() -> const Values& { return raw->getCurrentValues(); };
        fn_setCurrentValues_ = [raw](const Values& v) { raw->setCurrentValues(v); };
        fn_getOdomIds_ = [raw]() -> FastVector<uint64_t> { return raw->getOdometryPoseEdgeIds(); };
        fn_currentRank_ = [raw]() -> size_t { return raw->currentRank(); };
        fn_buildGraphAt_ = [raw](size_t p) -> NonlinearFactorGraph {
            // Always query an *unweighted* graph from the underlying problem.
            Vector saved = raw->getCurrentWeights();
            raw->clearCurrentWeights();
            NonlinearFactorGraph g = raw->buildGraphAtLevel(p);
            raw->setCurrentWeights(saved);
            return g;
        };
        fn_solveWeighted_ = [raw](const Vector& w, size_t pMin, size_t pMax, size_t iter)
            -> std::optional<CertificateResults> {
            return raw->SolveWeighted(w, pMin, pMax, iter);
        };
        fn_export_ = [raw](const std::string& path) {
            raw->ExportData(path, raw->RoundSolutionS(), false);
        };

        initializeFromProblem();
    }

private:
    /// Common initialization after the adapter functions are bound.
    void initializeFromProblem() {
        currentRank_ = fn_currentRank_();
        nfg_ = fn_buildGraphAt_(currentRank_);  // unweighted at construction-time rank
        state_ = fn_getCurrentValues_();

        // Set known inliers from odometry edges by default. Callers can add more.
        FastVector<uint64_t> odom = fn_getOdomIds_();
        params_.setKnownInliers(odom);

        sanityCheckKnownInliersAndOutliers();

        weights_ = initializeWeightsFromKnownInliersAndOutliers();
        setInlierCostThresholdsAtProbability(0.99);

        result_ = CertifiableGncResult{};
        result_.iter_log_.reserve(params_.maxIterations);
    }

    void sanityCheckKnownInliersAndOutliers() const {
        std::vector<size_t> inconsistent;
        std::set_intersection(params_.knownInliers.begin(), params_.knownInliers.end(),
                              params_.knownOutliers.begin(), params_.knownOutliers.end(),
                              std::back_inserter(inconsistent));
        if (!inconsistent.empty()) {
            throw std::runtime_error(
                "CertifiableGncOptimizer: one or more measurements were marked as BOTH "
                "known inlier AND known outlier.");
        }
        for (uint64_t i : params_.knownInliers) {
            if (i >= nfg_.size()) {
                throw std::runtime_error(
                    "CertifiableGncOptimizer: knownInlier index out of range.");
            }
        }
        for (uint64_t i : params_.knownOutliers) {
            if (i >= nfg_.size()) {
                throw std::runtime_error(
                    "CertifiableGncOptimizer: knownOutlier index out of range.");
            }
        }
    }

    /// Refresh nfg_ and barcSq_ if the certifier has climbed to a higher rank.
    void refreshIfRankChanged() {
        size_t newRank = fn_currentRank_();
        if (newRank != currentRank_) {
            currentRank_ = newRank;
            nfg_ = fn_buildGraphAt_(currentRank_);
            setInlierCostThresholdsAtProbability(0.99);
        }
    }

public:
    void setInlierCostThresholds(const double inth) {
        barcSq_ = inth * Vector::Ones(nfg_.size());
    }
    void setInlierCostThresholds(const Vector& inthVec) { barcSq_ = inthVec; }

    void setInlierCostThresholdsAtProbability(const double alpha) {
        barcSq_ = Vector::Ones(nfg_.size());
        for (size_t k = 0; k < nfg_.size(); ++k) {
            if (nfg_[k]) {
                // 0.5 factor follows GTSAM's NoiseModelFactor::error convention.
                barcSq_[k] = 0.5 * Certifiable_Chi2inv(alpha, nfg_[k]->dim());
            }
        }
    }

    void setWeights(const Vector& w) {
        if (size_t(w.size()) != nfg_.size()) {
            throw std::runtime_error("CertifiableGncOptimizer::setWeights: size mismatch.");
        }
        weights_ = w;
    }

    void setOutputPath(const std::string& path) { outputFile_ = path; }

    const NonlinearFactorGraph& getFactors() const { return nfg_; }
    const Values& getState() const { return state_; }
    const CertifiableGncParamsT& getParams() const { return params_; }
    const Vector& getWeights() const { return weights_; }
    const Vector& getInlierCostThresholds() const { return barcSq_; }

    Vector initializeWeightsFromKnownInliersAndOutliers() const {
        Vector w = Vector::Ones(nfg_.size());
        for (uint64_t i : params_.knownOutliers) w[i] = 0.0;
        return w;
    }

    /**
     * @brief Run GNC with the certifiable solver in the inner loop.
     *
     * Each outer iteration: compute new weights from current residuals and mu,
     * call SolveWeighted to re-optimize, check convergence, increase mu.
     *
     * Cost evaluation uses the underlying problem's *weighted* graph
     * (PGO->getCurrentGraph() after SolveWeighted) instead of cloning factors
     * — this avoids requiring NoiseModelFactor::clone() on the lifted factors.
     */
    CertifiableGncResult CertifiableOptimize(size_t pMin, size_t pMax) {
        auto total_start_time = CFGStopwatch::tick();

        // Iteration 0: solve with the initial (all-inlier or user-supplied) weights.
        auto cr0 = fn_solveWeighted_(weights_, pMin, pMax, 0);
        if (!cr0) {
            throw std::runtime_error(
                "CertifiableGncOptimizer: initial SolveWeighted returned nullopt.");
        }
        refreshIfRankChanged();
        Values result = cr0->Qstar;
        state_ = result;
        double cost = fn_getCurrentGraph_().error(result);
        double prev_cost = cost;
        recordIteration(0, cr0->total_computation_time, cr0->total_computation_time,
                        cr0->eta, cr0->verification_status_, cost, weights_, *cr0);
        if (params_.verbosity >= CertifiableGncParamsT::SUMMARY) {
            std::cout << "[GNC] iter=0 done  status=" << cr0->verification_status_
                      << "  cost=" << cost
                      << "  rank=" << cr0->endingRank
                      << "  time=" << cr0->total_computation_time << "s\n";
        }

        double mu = initializeMu();
        size_t unknownInOrOut = nfg_.size() - (params_.knownInliers.size() + params_.knownOutliers.size());
        if (mu <= 0 || unknownInOrOut == 0) {
            if (params_.verbosity >= CertifiableGncParamsT::SUMMARY) {
                std::cout << "[GNC] degenerate init (mu=" << mu
                          << ", unknownInOrOut=" << unknownInOrOut << "); skipping outer loop.\n";
            }
        } else {
            size_t iter;
            for (iter = 1; iter <= params_.maxIterations; ++iter) {
                if (params_.verbosity >= CertifiableGncParamsT::MU) {
                    std::cout << "[GNC] iter=" << iter << "  mu=" << mu << "\n";
                }

                auto outer_start = CFGStopwatch::tick();
                weights_ = calculateWeights(result, mu);

                if (params_.verbosity >= CertifiableGncParamsT::WEIGHTS) {
                    std::cout << "[GNC] weights min/max=" << weights_.minCoeff()
                              << "/" << weights_.maxCoeff() << "\n";
                }

                auto solve_start = CFGStopwatch::tick();
                auto cri = fn_solveWeighted_(weights_, pMin, pMax, iter);
                auto solve_end = CFGStopwatch::tock(solve_start);
                if (!cri) {
                    throw std::runtime_error(
                        "CertifiableGncOptimizer: SolveWeighted returned nullopt at iter "
                        + std::to_string(iter));
                }
                refreshIfRankChanged();
                result = cri->Qstar;
                state_ = result;

                cost = fn_getCurrentGraph_().error(result);

                bool done = checkConvergence(mu, weights_, cost, prev_cost);
                mu = updateMu(mu);
                auto outer_end = CFGStopwatch::tock(outer_start);
                recordIteration(iter, outer_end, solve_end, cri->eta,
                                cri->verification_status_, cost, weights_, *cri);

                if (params_.verbosity >= CertifiableGncParamsT::SUMMARY) {
                    std::cout << "[GNC] iter=" << iter << " done  status="
                              << cri->verification_status_
                              << "  cost=" << cost
                              << "  rank=" << cri->endingRank
                              << "  time=" << outer_end << "s"
                              << (done ? "  [converged]" : "") << "\n";
                }

                if (done) break;
                prev_cost = cost;
            }
        }

        auto total_end_time = CFGStopwatch::tock(total_start_time);
        result_.finalTotalTime_ = total_end_time;
        result_.finalEstimates_ = result;
        result_.finalWeights_ = weights_;

        if (!outputFile_.empty()) {
            fn_export_(outputFile_);
        }
        return result_;
    }

    // ---------------- GNC math (TLS only) ----------------

    double initializeMu() const {
        // Remark 5 in the GNC paper. TLS: pick mu such that the surrogate is convex
        // at initialization. Returns -1 if all residuals already below threshold.
        double mu_init = std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < nfg_.size(); ++k) {
            if (!nfg_[k]) continue;
            double rk = nfg_[k]->error(state_);
            double denom = 2.0 * rk - barcSq_[k];
            if (denom > 0) {
                mu_init = std::min(mu_init, barcSq_[k] / denom);
            }
        }
        if (mu_init >= 0 && mu_init < 1e-6) mu_init = 1e-6;
        return (mu_init > 0 && !std::isinf(mu_init)) ? mu_init : -1.0;
    }

    double updateMu(double mu) const {
        // TLS: mu grows each outer iteration (cost approaches the original TLS for mu -> inf).
        return mu * params_.muStep;
    }

    bool checkMuConvergence(double /*mu*/) const {
        // TLS has no stopping condition on mu (it diverges to infinity).
        return false;
    }

    bool checkCostConvergence(double cost, double prev_cost) const {
        if (!std::isfinite(cost) || !std::isfinite(prev_cost)) return false;
        const double diff = std::fabs(cost - prev_cost);
        const double denom = std::max(std::fabs(prev_cost), 1e-12);
        const double rel = diff / denom;
        bool ok = (diff < params_.absoluteErrorTol) || (rel < params_.relativeCostTol);
        if (ok && params_.verbosity >= CertifiableGncParamsT::SUMMARY) {
            std::cout << "[GNC] cost-convergence: prev=" << prev_cost
                      << " curr=" << cost << " |delta|=" << diff << " rel=" << rel << "\n";
        }
        return ok;
    }

    bool checkWeightsConvergence(const Vector& w) const {
        for (Eigen::Index i = 0; i < w.size(); ++i) {
            if (std::fabs(w[i] - std::round(w[i])) > params_.weightsTol) return false;
        }
        if (params_.verbosity >= CertifiableGncParamsT::SUMMARY) {
            std::cout << "[GNC] weights converged to binary.\n";
        }
        return true;
    }

    bool checkConvergence(double mu, const Vector& w, double cost, double prev_cost) const {
        return checkCostConvergence(cost, prev_cost)
            || checkWeightsConvergence(w)
            || checkMuConvergence(mu);
    }

    /// Build a weighted copy of nfg_ by scaling each factor's information matrix.
    NonlinearFactorGraph makeWeightedGraph(const Vector& w) const {
        NonlinearFactorGraph g;
        g.resize(nfg_.size());
        for (size_t i = 0; i < nfg_.size(); ++i) {
            if (!nfg_[i]) continue;
            auto factor = boost::dynamic_pointer_cast<NoiseModelFactor>(nfg_.at(i));
            if (!factor) {
                throw std::runtime_error(
                    "CertifiableGncOptimizer::makeWeightedGraph: non-noise-model factor.");
            }
            auto nm = boost::dynamic_pointer_cast<noiseModel::Gaussian>(factor->noiseModel());
            if (!nm) {
                throw std::runtime_error(
                    "CertifiableGncOptimizer::makeWeightedGraph: non-Gaussian noise model.");
            }
            double wi = w[i];
            if (wi < 1e-12) wi = 1e-12;
            Matrix newInfo = wi * nm->information();
            auto newNm = noiseModel::Gaussian::Information(newInfo);
            g[i] = factor->cloneWithNewNoiseModel(newNm);
        }
        return g;
    }

    Vector calculateWeights(const Values& currentEstimate, double mu) {
        Vector w = initializeWeightsFromKnownInliersAndOutliers();

        std::vector<size_t> allW(nfg_.size());
        std::iota(allW.begin(), allW.end(), 0);
        std::vector<size_t> knownW;
        std::set_union(params_.knownInliers.begin(), params_.knownInliers.end(),
                       params_.knownOutliers.begin(), params_.knownOutliers.end(),
                       std::back_inserter(knownW));
        std::vector<size_t> unknownW;
        std::set_difference(allW.begin(), allW.end(),
                            knownW.begin(), knownW.end(),
                            std::back_inserter(unknownW));

        // TLS weight update (eq. 14 of the GNC paper).
        for (size_t k : unknownW) {
            if (!nfg_[k]) continue;
            const double u2k = nfg_[k]->error(currentEstimate);
            const double upper = (mu + 1.0) / mu * barcSq_[k];
            const double lower = mu / (mu + 1.0) * barcSq_[k];
            double wk = std::sqrt(barcSq_[k] * mu * (mu + 1.0) / u2k) - mu;
            if (u2k >= upper || wk < 0.0) {
                wk = 0.0;
            } else if (u2k <= lower || wk > 1.0) {
                wk = 1.0;
            }
            w[k] = wk;
        }
        return w;
    }

    void recordIteration(size_t iter, double total_sec, double solve_sec, double eta,
                         VerificationStatus status, double objective,
                         const Vector& w, const CertificateResults& cr) {
        result_.iter_log_.push_back(IterInfo{iter, total_sec, solve_sec, eta, status,
                                             objective, w, cr});
    }

    // ---------------- Logging ----------------

    /**
     * @brief Write per-iteration TXT + CSV log and the final weights vector.
     *
     * Files written under file_stem:
     *   <stem>_iter_log.txt      human-readable per-iter summary
     *   <stem>_iters.csv         per-iter scalars (iter, objective, times, ranks)
     *   <stem>_weights.txt       one row per factor: "index weight"
     *   <stem>_weights.csv       (optional, long-form: iter,edge_index,weight)
     * Appends one row per run to <export_path>/final_timings.csv.
     */
    void exportGncIterationLogTXTandCSVs(const CertifiableGncResult& res,
                                         const std::string& file_stem,
                                         bool dump_weights_csv = false,
                                         size_t max_weights_in_txt = 0) {
        {
            std::ofstream txt(file_stem + "_iter_log.txt");
            if (!txt) {
                std::cerr << "[export] cannot open: " << file_stem << "_iter_log.txt\n";
                return;
            }
            txt << std::setprecision(17);
            txt << "=== GNC RUN ===\n";
            txt << "num_iters: " << res.iter_log_.size() << "\n";
            txt << "final_total_time: " << res.finalTotalTime_ << "\n";
            txt << "final_weights_size: " << res.finalWeights_.size() << "\n\n";

            for (size_t it = 0; it < res.iter_log_.size(); ++it) {
                const IterInfo& I = res.iter_log_[it];
                const auto& R = I.certifiableResults_;
                double opt_sum = std::accumulate(R.elapsed_optimization_times.begin(),
                                                 R.elapsed_optimization_times.end(), 0.0);
                txt << "---- ITER " << (it + 1) << " ----\n";
                txt << "objective: " << I.objective_ << "\n";
                txt << "out_iter_time: " << I.out_iter_time_ << "\n";
                txt << "solve_time: " << I.solve_time_ << "\n";
                txt << "eta: " << I.eta_ << "\n";
                txt << "Verification status: " << I.verification_status_ << "\n";
                txt << "weights_size: " << I.IterInfo_Weights_.size() << "\n";
                if (max_weights_in_txt > 0 && I.IterInfo_Weights_.size() > 0) {
                    auto nshow = std::min<Eigen::Index>(I.IterInfo_Weights_.size(),
                                                        static_cast<Eigen::Index>(max_weights_in_txt));
                    txt << "weights_preview(count=" << nshow << "):";
                    for (Eigen::Index k = 0; k < nshow; ++k) {
                        txt << " [" << k << "]=" << I.IterInfo_Weights_(k);
                    }
                    txt << "\n";
                }
                txt << "cert.startingRank: " << R.startingRank << "\n";
                txt << "cert.endingRank: " << R.endingRank << "\n";
                txt << "cert.total_computation_time: " << R.total_computation_time << "\n";
                txt << "cert.sum_optimization_time: " << opt_sum << "\n\n";
            }
            if (res.finalWeights_.size() > 0) {
                const double eps = 1e-3;
                size_t nnz = 0;
                for (Eigen::Index k = 0; k < res.finalWeights_.size(); ++k) {
                    if (res.finalWeights_(k) > eps) ++nnz;
                }
                txt << "=== FINAL ===\n";
                txt << "final_total_time: " << res.finalTotalTime_ << "\n";
                txt << "final_weights_size: " << res.finalWeights_.size() << "\n";
                txt << "final_weights_min: " << res.finalWeights_.minCoeff() << "\n";
                txt << "final_weights_max: " << res.finalWeights_.maxCoeff() << "\n";
                txt << "final_weights_nnz(> " << eps << "): " << nnz << "\n";
            }
        }

        {
            std::ofstream csv(file_stem + "_iters.csv");
            if (!csv) return;
            csv << std::setprecision(17);
            csv << "iter,objective,out_iter_time,solve_time,cert_startingRank,cert_endingRank,"
                   "cert_total_computation_time,opt_sum_time\n";
            for (size_t it = 0; it < res.iter_log_.size(); ++it) {
                const IterInfo& I = res.iter_log_[it];
                const auto& R = I.certifiableResults_;
                double opt_sum = std::accumulate(R.elapsed_optimization_times.begin(),
                                                 R.elapsed_optimization_times.end(), 0.0);
                csv << (it + 1) << "," << I.objective_ << "," << I.out_iter_time_ << ","
                    << I.solve_time_ << "," << R.startingRank << "," << R.endingRank << ","
                    << R.total_computation_time << "," << opt_sum << "\n";
            }
        }

        if (dump_weights_csv) {
            std::ofstream wcsv(file_stem + "_weights.csv");
            if (!wcsv) return;
            wcsv << std::setprecision(17);
            wcsv << "iter,edge_index,weight\n";
            for (size_t it = 0; it < res.iter_log_.size(); ++it) {
                const auto& W = res.iter_log_[it].IterInfo_Weights_;
                for (Eigen::Index k = 0; k < W.size(); ++k) {
                    wcsv << (it + 1) << "," << k << "," << W(k) << "\n";
                }
            }
        }

        if (res.finalWeights_.size() > 0) {
            std::ofstream wtxt(file_stem + "_weights.txt");
            if (wtxt) {
                wtxt << std::setprecision(17);
                wtxt << "# final_weights (size=" << res.finalWeights_.size() << ")\n";
                for (Eigen::Index k = 0; k < res.finalWeights_.size(); ++k) {
                    wtxt << k << " " << res.finalWeights_(k) << "\n";
                }
            }
        }

        // ---------- Per-run summary CSV (Monte-Carlo friendly) ----------
        // Wall-clock metrics come from the FINAL iter's certificate vectors
        // (CertifiableProblem accumulates across SolveWeighted calls).
        // Rank-climbing metrics come from per-iter endingRank diffs.
        {
            std::ofstream summ(file_stem + "_summary.csv");
            if (summ) {
                summ << std::setprecision(17);
                size_t num_iters = res.iter_log_.size();
                size_t max_end = 0;
                size_t levels_total = 0;
                size_t n_ver = 0, n_fail = 0;
                size_t final_start = 0, final_end = 0;
                double final_obj = std::numeric_limits<double>::quiet_NaN();
                double final_eta = std::numeric_limits<double>::quiet_NaN();
                size_t prev_end = 0;
                for (size_t it = 0; it < num_iters; ++it) {
                    const auto& I = res.iter_log_[it];
                    const auto& R = I.certifiableResults_;
                    max_end = std::max(max_end, R.endingRank);
                    // Per-iter levels climbed = endingRank - startingRank where
                    // startingRank is the rank the certifier was reset to at the
                    // start of this SolveWeighted call (always pMin in the wrapper).
                    if (R.endingRank > R.startingRank) {
                        levels_total += (R.endingRank - R.startingRank);
                    }
                    if (I.verification_status_ == VerificationStatus::VERIFIED) ++n_ver;
                    else if (I.verification_status_ == VerificationStatus::FAILED) ++n_fail;
                    final_start = R.startingRank;
                    final_end = R.endingRank;
                    final_obj = I.objective_;
                    final_eta = I.eta_;
                    prev_end = R.endingRank;
                }
                // Read the final iter's cumulative time vectors (these are
                // append-only over the lifetime of the certifier object).
                double sum_lm = 0.0, sum_ver = 0.0, sum_init = 0.0, sum_cert = 0.0;
                if (!res.iter_log_.empty()) {
                    const auto& Rf = res.iter_log_.back().certifiableResults_;
                    for (double t : Rf.elapsed_optimization_times) sum_lm += t;
                    for (double t : Rf.verification_times) sum_ver += t;
                    for (double t : Rf.initialization_time) sum_init += t;
                    sum_cert = Rf.total_computation_time;
                }
                summ << "field,value\n";
                summ << "method,certifiable\n";
                summ << "num_gnc_iters," << num_iters << "\n";
                summ << "final_starting_rank," << final_start << "\n";
                summ << "final_ending_rank," << final_end << "\n";
                summ << "max_ending_rank," << max_end << "\n";
                summ << "levels_climbed_total," << levels_total << "\n";
                summ << "num_certifier_verified," << n_ver << "\n";
                summ << "num_certifier_failed," << n_fail << "\n";
                summ << "sum_lm_opt_time," << sum_lm << "\n";
                summ << "sum_verification_time," << sum_ver << "\n";
                summ << "sum_initialization_time," << sum_init << "\n";
                summ << "sum_certifier_time," << sum_cert << "\n";
                summ << "final_objective," << final_obj << "\n";
                summ << "final_eta," << final_eta << "\n";
                summ << "final_total_time," << res.finalTotalTime_ << "\n";
                (void)prev_end;
            }
        }

        // Monte-Carlo timing CSV (append).
        try {
            if (!params_.export_path.empty()) {
                std::filesystem::path dir(params_.export_path);
                std::error_code ec;
                std::filesystem::create_directories(dir, ec);
                std::filesystem::path csvp = dir / "final_timings.csv";
                bool need_header = !std::filesystem::exists(csvp)
                                   || std::filesystem::file_size(csvp) == 0;
                std::ofstream mc(csvp, std::ios::app);
                if (mc) {
                    mc << std::setprecision(17);
                    if (need_header) mc << "run_id,final_total_time_s\n";
                    mc << std::filesystem::path(file_stem).filename().string()
                       << "," << res.finalTotalTime_ << "\n";
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[export] exception writing final_timings.csv: " << e.what() << "\n";
        }
    }
};

}  // namespace gtsam

#endif  // STIEFELMANIFOLDEXAMPLE_CERTIFIABLEGNCOPTIMIZER_H
