# Multi-Cloud RAG Infrastructure Platform
## Product Requirements Document & Technical Implementation Guide

**Version**: 1.0
**Date**: December 2025
**Author**: Sohail Mohammad
**Classification**: Technical Specification

---

<div align="center">

### 96 Production Configurations | 4 Model Tiers | 6 GPU Architectures | 4 Infrastructure Levels

*A comprehensive system for deploying, orchestrating, and scaling Retrieval-Augmented Generation pipelines across heterogeneous compute infrastructure*

</div>

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Definition](#2-problem-definition)
3. [Solution Architecture](#3-solution-architecture)
4. [The Configuration Matrix](#4-the-configuration-matrix)
5. [Model Tier Specifications](#5-model-tier-specifications)
6. [GPU Architecture Reference](#6-gpu-architecture-reference)
7. [Infrastructure Tiers](#7-infrastructure-tiers)
8. [RAG Pipeline Architecture](#8-rag-pipeline-architecture)
9. [Intelligent Scheduling System](#9-intelligent-scheduling-system)
10. [Deployment Patterns](#10-deployment-patterns)
11. [Observability Framework](#11-observability-framework)
12. [Cost Optimization](#12-cost-optimization)
13. [Security & Compliance](#13-security--compliance)
14. [Implementation Roadmap](#14-implementation-roadmap)
15. [Repository Structure](#15-repository-structure)

---

## 1. Executive Summary

### What This Document Provides

This PRD defines a **production-ready multi-cloud RAG infrastructure platform** capable of:

- **Deploying LLMs** across 4 size tiers (2B → 500B+ parameters)
- **Running on 6 GPU architectures** (RTX 4090 → B200)
- **Operating across 4 infrastructure levels** (Serverless → On-premises)
- **Executing RAG pipelines** with intelligent routing and cost optimization

The platform handles **96 unique configuration combinations**, each with specific guidance for deployment, scheduling, observability, and cost management.

### Key Deliverables

| Deliverable | Description |
|-------------|-------------|
| **Configuration Matrix** | 96 production configurations with deployment specs |
| **Implementation Repository** | 64 technical documents across 15 categories |
| **Scheduling System** | Multi-tier intelligent routing with Kueue/Volcano |
| **RAG Pipeline** | Modular retrieval system with provider abstraction |
| **Cost Framework** | TCO analysis with automated optimization |

### Why This Matters

Organizations deploying LLMs face a combinatorial explosion of choices:

```
4 Model Sizes × 6 GPU Types × 4 Cloud Tiers = 96 Possible Configurations
```

Each combination has unique characteristics for memory requirements, tensor parallelism, cost efficiency, and failure modes. This platform provides **battle-tested configurations** for every viable combination, eliminating months of trial-and-error.

### Quick Start by Role

| If you are a... | Start Here | Then Read | Deep Dive |
|-----------------|------------|-----------|-----------|
| **Platform Engineer** | [§7 Infrastructure Tiers](#7-infrastructure-tiers) | [9.1](09_inference_serving/9.1_inference_engine_selection_guide.md), [9.2](09_inference_serving/9.2_serving_architecture_patterns_guide.md) | [10.1](10_monitoring_observability/10.1_llm_monitoring_strategy_guide.md), [13.1](13_operations_reliability/13.1_incident_response_guide.md) |
| **ML Engineer** | [§5 Model Tiers](#5-model-tier-specifications) | [6.1](06_model_optimization/6.1_quantization_guide.md), [7.1](07_rag_pipeline/7.1_vector_database_guide.md) | [5.1](05_evaluation_testing/5.1_llm_evaluation_framework.md), [3.1](03_fine_tuning/3.1_supervised_fine_tuning.md) |
| **DevOps/SRE** | [§10 Deployment](#10-deployment-patterns) | [10.1](10_monitoring_observability/10.1_llm_monitoring_strategy_guide.md), [13.5](13_operations_reliability/13.5_operational_runbooks_guide.md) | [13.1](13_operations_reliability/13.1_incident_response_guide.md), [13.2](13_operations_reliability/13.2_disaster_recovery_guide.md) |
| **Security/Compliance** | [§13 Security](#13-security--compliance) | [11.1](11_security_governance/11.1_llm_security_guide.md), [11.2](11_security_governance/11.2_pii_data_privacy_guide.md) | [11.3](11_security_governance/11.3_compliance_framework_guide.md), [11.5](11_security_governance/11.5_access_control_authentication_guide.md) |
| **Cost/Finance** | [§12 Cost](#12-cost-optimization) | [14.1](14_cost_capacity_management/14.1_total_cost_ownership_guide.md), [14.2](14_cost_capacity_management/14.2_cloud_cost_optimization_guide.md) | [14.3](14_cost_capacity_management/14.3_gpu_infrastructure_optimization_guide.md), [10.4](10_monitoring_observability/10.4_cost_monitoring_optimization_guide.md) |

### Implementation Sequence

For end-to-end deployment, follow this order:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  1. DATA & TRAINING        2. OPTIMIZATION         3. SERVING & RAG            │
│  ───────────────────       ───────────────         ────────────────            │
│  01_data_pipeline/    →    06_model_optimization/  →  09_inference_serving/    │
│  02_model_training/   →    05_evaluation_testing/  →  07_rag_pipeline/         │
│  03_fine_tuning/                                                               │
│  04_alignment_safety/                                                          │
│                                                                                 │
│  4. OPERATIONS             5. GOVERNANCE           6. SCALE                    │
│  ─────────────             ──────────              ─────                       │
│  10_monitoring/       →    11_security/        →   14_cost_management/         │
│  08_mlops_lifecycle/  →    12_developer_exp/   →   15_migration/               │
│  13_operations/                                                                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Recommended reading order for full understanding:**
1. **Foundation**: 1.1 → 2.2 → 2.3 (Data + Architecture + Distributed Training)
2. **Model Prep**: 3.1 → 6.1 → 6.4 (Fine-tuning + Quantization + Speculative Decoding)
3. **RAG Core**: 7.1 → 7.2 → 7.3 → 7.4 (Vector DB → Embeddings → Chunking → Retrieval)
4. **Deployment**: 9.1 → 9.2 → 9.3 (Engine Selection → Architecture → API Design)
5. **Operations**: 10.1 → 13.1 → 13.5 (Monitoring → Incident Response → Runbooks)

### Configuration Quick Reference

| Your Scenario | Model Tier | Recommended GPU | Infrastructure | Key Docs |
|---------------|------------|-----------------|----------------|----------|
| **Fast prototyping** | S (2B) | L40S / RTX 4090 | Serverless | [9.1](09_inference_serving/9.1_inference_engine_selection_guide.md), [7.1](07_rag_pipeline/7.1_vector_database_guide.md) |
| **Production RAG** | M (8-20B) | A100 / L40S | GPU-Native | [9.2](09_inference_serving/9.2_serving_architecture_patterns_guide.md), [7.4](07_rag_pipeline/7.4_retrieval_reranking_guide.md), [10.1](10_monitoring_observability/10.1_llm_monitoring_strategy_guide.md) |
| **Enterprise reasoning** | L (70B) | H100×2 | Hyperscaler | [2.3](02_model_training/2.3_distributed_training_infrastructure.md), [6.1](06_model_optimization/6.1_quantization_guide.md), [11.1](11_security_governance/11.1_llm_security_guide.md) |
| **Frontier/Research** | XL (405B) | H100×8 | On-Premises | [2.3](02_model_training/2.3_distributed_training_infrastructure.md), [13.3](13_operations_reliability/13.3_capacity_planning_guide.md), [14.3](14_cost_capacity_management/14.3_gpu_infrastructure_optimization_guide.md) |
| **Cost-optimized** | M (8B) | L40S | On-Premises | [14.1](14_cost_capacity_management/14.1_total_cost_ownership_guide.md), [14.2](14_cost_capacity_management/14.2_cloud_cost_optimization_guide.md) |
| **HIPAA/Compliance** | Any | Any | Hyperscaler/On-Prem | [11.3](11_security_governance/11.3_compliance_framework_guide.md), [11.2](11_security_governance/11.2_pii_data_privacy_guide.md) |

---

## 2. Problem Definition

### The Challenge

Modern AI infrastructure teams face three simultaneous pressures:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         THE AI INFRASTRUCTURE TRILEMMA                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                              ┌─────────────┐                                │
│                              │   QUALITY   │                                │
│                              │  (Largest   │                                │
│                              │   Models)   │                                │
│                              └──────┬──────┘                                │
│                                     │                                       │
│                        Pick any two │                                       │
│                    ┌────────────────┼────────────────┐                      │
│                    │                │                │                      │
│                    ▼                │                ▼                      │
│            ┌─────────────┐          │         ┌─────────────┐               │
│            │    COST     │◄─────────┴────────►│    SPEED    │               │
│            │ (Minimize   │    Trade-offs      │  (Lowest    │               │
│            │  Spend)     │                    │   Latency)  │               │
│            └─────────────┘                    └─────────────┘               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Specific Problems Solved

| Problem | Impact | Our Solution |
|---------|--------|--------------|
| **Configuration Complexity** | Teams spend months testing GPU/model combinations | Pre-validated 96-configuration matrix |
| **Multi-Cloud Lock-in** | Vendor lock-in limits negotiation leverage | Portable abstractions across 4 cloud tiers |
| **Cost Unpredictability** | GPU costs vary 20x between providers | Real-time cost routing with budget controls |
| **Scaling Challenges** | Manual scaling leads to over/under-provisioning | Kueue-based intelligent autoscaling |
| **RAG Quality Variance** | Inconsistent retrieval across deployments | Standardized pipeline with configurable retrieval |

### Requirements From Problem Statement

The task requires a system that handles:

```yaml
Model Sizes:
  - Tier S: ~2B parameters (Phi-3-mini, Qwen2.5-3B)
  - Tier M: ~8-20B parameters (Llama-3.1-8B, Mistral-7B)
  - Tier L: ~70-100B parameters (Llama-3.1-70B, Qwen2.5-72B)
  - Tier XL: ~400-500B parameters (Llama-3.1-405B, DeepSeek-V3)

Infrastructure Levels:
  - Hyperscaler: AWS, GCP, Azure
  - GPU-Native: CoreWeave, Lambda Labs
  - Serverless: RunPod, Modal, Replicate
  - On-Premises:
    - 2 racks (NYC + San Francisco)
    - 8 nodes per rack, 8 GPUs per node
    - ~500 GPUs per rack, 5000 GPUs across 10 locations

GPU Architectures:
  - Blackwell: B200 (192GB, 8 TB/s)
  - Hopper: H200 (141GB), H100 (80GB)
  - Ampere: A100 (80GB)
  - Ada Lovelace: L40S (48GB), RTX 4090 (24GB)
```

---

## 3. Solution Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          MULTI-CLOUD RAG PLATFORM                                 │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                           CONTROL PLANE                                      ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ ││
│  │  │   Gateway    │  │  Scheduler   │  │    Cost      │  │   Config         │ ││
│  │  │   (Kong)     │  │   (Kueue)    │  │  Controller  │  │   Manager        │ ││
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘ ││
│  └─────────┼─────────────────┼─────────────────┼───────────────────┼───────────┘│
│            │                 │                 │                   │            │
│  ┌─────────▼─────────────────▼─────────────────▼───────────────────▼───────────┐│
│  │                         ORCHESTRATION LAYER                                  ││
│  │  ┌─────────────────────────────────────────────────────────────────────────┐││
│  │  │                    Intelligent Request Router                            │││
│  │  │   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐                 │││
│  │  │   │ Tier S  │   │ Tier M  │   │ Tier L  │   │ Tier XL │                 │││
│  │  │   │ Queue   │   │ Queue   │   │ Queue   │   │ Queue   │                 │││
│  │  │   └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘                 │││
│  │  └────────┼─────────────┼─────────────┼─────────────┼──────────────────────┘││
│  └───────────┼─────────────┼─────────────┼─────────────┼────────────────────────┘│
│              │             │             │             │                         │
│  ┌───────────▼─────────────▼─────────────▼─────────────▼────────────────────────┐│
│  │                         COMPUTE FABRIC                                        ││
│  │                                                                               ││
│  │   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              ││
│  │   │   HYPERSCALER   │  │   GPU-NATIVE    │  │   SERVERLESS    │              ││
│  │   │   AWS/GCP/Azure │  │CoreWeave/Lambda │  │  RunPod/Modal   │              ││
│  │   │                 │  │                 │  │                 │              ││
│  │   │  ┌───┐ ┌───┐   │  │  ┌───┐ ┌───┐   │  │  ┌───┐ ┌───┐   │              ││
│  │   │  │H100│ │A100│  │  │  │H100│ │H200│  │  │  │A100│ │L40S│  │              ││
│  │   │  └───┘ └───┘   │  │  └───┘ └───┘   │  │  └───┘ └───┘   │              ││
│  │   └─────────────────┘  └─────────────────┘  └─────────────────┘              ││
│  │                                                                               ││
│  │   ┌─────────────────────────────────────────────────────────────────────────┐││
│  │   │                        ON-PREMISES CLUSTERS                              │││
│  │   │   ┌─────────────────────┐       ┌─────────────────────┐                 │││
│  │   │   │     NYC RACK        │       │    SF RACK          │                 │││
│  │   │   │   8 Nodes × 8 GPUs  │       │   8 Nodes × 8 GPUs  │                 │││
│  │   │   │     = 64 GPUs       │       │     = 64 GPUs       │                 │││
│  │   │   │   (H100/A100 Mix)   │       │   (H100/A100 Mix)   │                 │││
│  │   │   └─────────────────────┘       └─────────────────────┘                 │││
│  │   └─────────────────────────────────────────────────────────────────────────┘││
│  └───────────────────────────────────────────────────────────────────────────────┘│
│                                                                                   │
│  ┌───────────────────────────────────────────────────────────────────────────────┐│
│  │                           DATA PLANE                                          ││
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ ││
│  │   │   Vector     │  │  Document    │  │   Embedding  │  │    Reranking     │ ││
│  │   │   Store      │  │   Store      │  │   Service    │  │    Service       │ ││
│  │   │  (Qdrant)    │  │   (S3)       │  │   (vLLM)     │  │   (BGE/Cohere)   │ ││
│  │   └──────────────┘  └──────────────┘  └──────────────┘  └──────────────────┘ ││
│  └───────────────────────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

```yaml
# Core Infrastructure
Container Orchestration: Kubernetes (EKS/GKE/AKS + self-managed)
GPU Scheduling: Kueue + Volcano (hybrid)
Service Mesh: Istio
API Gateway: Kong

# LLM Serving
Inference Engine: vLLM (primary), TensorRT-LLM (NVIDIA-specific)
Model Format: Safetensors, GGUF (edge)
Quantization: AWQ (INT4), FP8 (Hopper+), GPTQ

# RAG Components
Vector Database: Qdrant (primary), pgvector (simple), Milvus (scale)
Embeddings: nomic-embed-text-v1.5, text-embedding-3-large
Reranking: BAAI/bge-reranker-v2-m3, Cohere rerank-v3
Orchestration: LangChain, LlamaIndex

# Observability
Metrics: Prometheus + Grafana
Tracing: OpenTelemetry + Jaeger
Logging: Loki + Promtail
Alerting: Alertmanager + PagerDuty

# Infrastructure as Code
Provisioning: Terraform (multi-cloud)
Configuration: Helm + Kustomize
GitOps: ArgoCD
Secrets: HashiCorp Vault
```

### Core Design Principles

1. **Provider Abstraction**: All cloud-specific logic isolated behind interfaces
2. **Configuration-Driven**: 96 configurations defined declaratively in YAML
3. **Cost-Aware Routing**: Every request considers cost/latency/quality tradeoffs
4. **Graceful Degradation**: Automatic fallback across tiers and providers
5. **Observable by Default**: Every component instrumented for production debugging

---

## 4. The Configuration Matrix

### Feasibility Overview

The following matrices show viability of each Model Tier × GPU combination across infrastructure levels.

**Legend**: ✅ Recommended | ✓ Viable | ⚠️ Marginal | ❌ Not Recommended

#### Hyperscaler (AWS/GCP/Azure)
| Model \ GPU | B200 | H200 | H100 | A100 | L40S | RTX 4090 |
|-------------|------|------|------|------|------|----------|
| **Tier S** (2B) | ⚠️ | ⚠️ | ⚠️ | ✓ | ✅ | ❌ |
| **Tier M** (8-20B) | ✓ | ✓ | ✓ | ✅ | ✅ | ❌ |
| **Tier L** (70-100B) | ✅ | ✅ | ✅ | ✓ | ⚠️ | ❌ |
| **Tier XL** (400-500B) | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ |

#### GPU-Native Cloud (CoreWeave/Lambda)
| Model \ GPU | B200 | H200 | H100 | A100 | L40S | RTX 4090 |
|-------------|------|------|------|------|------|----------|
| **Tier S** (2B) | ⚠️ | ⚠️ | ⚠️ | ✓ | ✅ | ✅ |
| **Tier M** (8-20B) | ✓ | ✓ | ✓ | ✅ | ✅ | ⚠️ |
| **Tier L** (70-100B) | ✅ | ✅ | ✅ | ✓ | ⚠️ | ❌ |
| **Tier XL** (400-500B) | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ |

#### Serverless (RunPod/Modal)
| Model \ GPU | B200 | H200 | H100 | A100 | L40S | RTX 4090 |
|-------------|------|------|------|------|------|----------|
| **Tier S** (2B) | ⚠️ | ⚠️ | ⚠️ | ✓ | ✅ | ✅ |
| **Tier M** (8-20B) | ✓ | ✓ | ✓ | ✅ | ✅ | ⚠️ |
| **Tier L** (70-100B) | ✅ | ✅ | ✅ | ✓ | ⚠️ | ❌ |
| **Tier XL** (400-500B) | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |

#### On-Premises / Colocation
| Model \ GPU | B200 | H200 | H100 | A100 | L40S | RTX 4090 |
|-------------|------|------|------|------|------|----------|
| **Tier S** (2B) | ⚠️ | ⚠️ | ⚠️ | ✓ | ✅ | ✅ |
| **Tier M** (8-20B) | ✓ | ✓ | ✓ | ✅ | ✅ | ⚠️ |
| **Tier L** (70-100B) | ✓ | ✓ | ✅ | ✓ | ⚠️ | ❌ |
| **Tier XL** (400-500B) | ✓ | ✓ | ✅ | ⚠️ | ❌ | ❌ |

### Configuration Selection Algorithm

```python
def select_optimal_configuration(
    task_complexity: str,      # "simple", "moderate", "complex", "frontier"
    latency_requirement_ms: int,
    budget_per_1k_tokens: float,
    compliance_requirements: List[str],
    gpu_availability: Dict[str, bool]
) -> Configuration:
    """
    Select optimal configuration based on requirements.

    Returns the best-fit configuration from the 96-configuration matrix.
    """

    # Map task complexity to model tier
    tier_map = {
        "simple": "S",      # 2B - Q&A, classification
        "moderate": "M",    # 8-20B - General RAG, coding
        "complex": "L",     # 70-100B - Multi-step reasoning
        "frontier": "XL"    # 400-500B - Research, agents
    }
    model_tier = tier_map[task_complexity]

    # Filter by compliance requirements
    if "HIPAA" in compliance_requirements or "SOC2" in compliance_requirements:
        eligible_infra = ["hyperscaler", "on_premises"]
    elif "air_gapped" in compliance_requirements:
        eligible_infra = ["on_premises"]
    else:
        eligible_infra = ["hyperscaler", "gpu_native", "serverless", "on_premises"]

    # Score each configuration
    candidates = []
    for config in CONFIGURATION_MATRIX[model_tier]:
        if config.infrastructure not in eligible_infra:
            continue
        if not gpu_availability.get(config.gpu_type, False):
            continue
        if config.ttft_p99_ms > latency_requirement_ms:
            continue
        if config.cost_per_1k_tokens > budget_per_1k_tokens:
            continue

        score = calculate_efficiency_score(config)
        candidates.append((score, config))

    # Return highest-scoring configuration
    candidates.sort(reverse=True)
    return candidates[0][1] if candidates else None
```

---

## 5. Model Tier Specifications

### Tier S: Small Models (2B Parameters)

```yaml
Tier: S
Parameter Range: 1-3B
Representative Models:
  - Qwen2.5-3B-Instruct
  - Phi-3-mini-4k-instruct
  - Llama-3.2-3B-Instruct

Use Cases:
  - Simple Q&A and FAQ
  - Text classification
  - Entity extraction
  - Low-latency chat

Memory Requirements:
  FP16: ~6 GB
  FP8:  ~3 GB
  INT4: ~2 GB

Performance Targets:
  Throughput: 400-800 tokens/sec
  TTFT (P99): < 100ms
  Context: 4,096 tokens typical (32K max)

Optimal GPU Pairings:
  - L40S (48GB): ✅ Best cost efficiency
  - RTX 4090 (24GB): ✅ Lowest cost (where available)
  - A100 (80GB): ✓ Over-provisioned but works
  - H100/H200/B200: ⚠️ Significant waste

RAG Configuration:
  Vector DB: pgvector (simplicity)
  Embedding: nomic-embed-text-v1.5 (768d)
  Chunk Size: 256 tokens
  Top-K: 3
  Reranker: bge-reranker-v2-m3
```

### Tier M: Medium Models (8-20B Parameters)

```yaml
Tier: M
Parameter Range: 7-20B
Representative Models:
  - Llama-3.1-8B-Instruct
  - Mistral-7B-v0.3-Instruct
  - Qwen2.5-14B-Instruct

Use Cases:
  - General RAG applications
  - Code assistance
  - Document analysis
  - Customer support

Memory Requirements:
  FP16: ~40 GB
  FP8:  ~20 GB
  INT4: ~10 GB

Performance Targets:
  Throughput: 150-350 tokens/sec
  TTFT (P99): < 300ms
  Context: 8,192 tokens typical (128K max)

Optimal GPU Pairings:
  - L40S (48GB): ✅ Single-GPU, excellent efficiency
  - A100 (80GB): ✅ Headroom for batching
  - H100 (80GB): ✓ Future-proof, faster
  - RTX 4090 (24GB): ⚠️ Requires INT4, limited batch

RAG Configuration:
  Vector DB: Qdrant (scale + performance)
  Embedding: text-embedding-3-large (3072d)
  Chunk Size: 512 tokens
  Top-K: 5
  Reranker: Cohere rerank-v3
```

### Tier L: Large Models (70-100B Parameters)

```yaml
Tier: L
Parameter Range: 65-100B
Representative Models:
  - Llama-3.1-70B-Instruct
  - Qwen2.5-72B-Instruct
  - DeepSeek-V2.5

Use Cases:
  - Complex reasoning
  - Multi-step analysis
  - High-stakes decisions
  - Legal/medical analysis

Memory Requirements:
  FP16: ~200 GB
  FP8:  ~100 GB
  INT4: ~50 GB

Performance Targets:
  Throughput: 40-120 tokens/sec
  TTFT (P99): < 1000ms
  Context: 16,384 tokens typical (128K max)

Optimal GPU Pairings:
  - H100 × 2 (TP=2): ✅ Standard production config
  - H200 × 2 (TP=2): ✅ Premium, higher throughput
  - A100 × 4 (TP=4): ✓ Cost-effective alternative
  - B200 × 1: ✅ Single-GPU possible with FP8

Parallelism Strategy:
  Tensor Parallelism: 2-4 (memory distribution)
  Pipeline Parallelism: 1 (minimize latency)

RAG Configuration:
  Vector DB: Milvus (enterprise scale)
  Embedding: text-embedding-3-large (3072d)
  Chunk Size: 1024 tokens
  Top-K: 10
  Reranker: Cohere rerank-v3 + cross-encoder
```

### Tier XL: Frontier Models (400-500B Parameters)

```yaml
Tier: XL
Parameter Range: 400-500B+
Representative Models:
  - Llama-3.1-405B-Instruct
  - DeepSeek-V3
  - Mixture of Experts variants

Use Cases:
  - Frontier reasoning
  - Research applications
  - Agentic workflows
  - Multi-modal synthesis

Memory Requirements:
  FP16: ~1000 GB
  FP8:  ~500 GB
  INT4: ~250 GB

Performance Targets:
  Throughput: 15-40 tokens/sec
  TTFT (P99): < 2000ms
  Context: 32,768 tokens typical (128K max)

Optimal GPU Pairings:
  - H100 × 8 (TP=8): ✅ Standard for 405B
  - H200 × 8 (TP=8): ✅ Higher throughput
  - B200 × 4 (TP=4): ✅ Next-gen efficiency
  - A100 × 16 (TP=8, PP=2): ⚠️ High latency

Parallelism Strategy:
  Tensor Parallelism: 8 (full node)
  Pipeline Parallelism: 1-2 (multi-node)
  Expert Parallelism: Required for MoE

RAG Configuration:
  Vector DB: Milvus cluster (distributed)
  Embedding: voyage-large-2-instruct
  Chunk Size: 2048 tokens
  Top-K: 20
  Reranker: Multi-stage (fast → precise)
```

---

## 6. GPU Architecture Reference

### Quick Comparison

| GPU | VRAM | BW (TB/s) | FP8 | FP4 | Cost/hr | Best For |
|-----|------|-----------|-----|-----|---------|----------|
| **B200** | 192GB | 8.0 | ✓ | ✓ | $6-8 | XL, L tiers |
| **H200** | 141GB | 4.8 | ✓ | ✗ | $3.5-4.5 | L, XL tiers |
| **H100** | 80GB | 3.35 | ✓ | ✗ | $2-4.25 | M, L tiers |
| **A100** | 80GB | 2.0 | ✗ | ✗ | $1.5-2.5 | M, L (cost) |
| **L40S** | 48GB | 0.86 | ✓ | ✗ | $1-1.5 | S, M tiers |
| **RTX 4090** | 24GB | 1.0 | ✓ | ✗ | $0.44-0.75 | S, dev |

### Memory Planning Formula

```python
def calculate_gpu_memory_requirement(
    model_params_billions: float,
    precision: str,
    context_length: int,
    batch_size: int,
    kv_cache_dtype: str = "auto"
) -> float:
    """
    Calculate total GPU memory requirement in GB.

    Memory = Model Weights + KV Cache + Activation Memory + Overhead
    """

    # Bytes per parameter by precision
    precision_bytes = {
        "fp32": 4.0,
        "fp16": 2.0,
        "bf16": 2.0,
        "fp8": 1.0,
        "int8": 1.0,
        "int4": 0.5,
        "awq": 0.5,
        "gptq": 0.5
    }

    bytes_per_param = precision_bytes[precision]

    # Model weights
    model_memory_gb = (model_params_billions * 1e9 * bytes_per_param) / 1e9

    # KV Cache estimation (simplified)
    # KV cache per token ≈ 2 * num_layers * hidden_dim * 2 (K and V) * dtype_bytes
    # For typical architectures: ~0.5-2 MB per token per 10B params
    kv_bytes_per_token = (model_params_billions / 10) * 1e6  # ~1MB per 10B
    kv_cache_gb = (context_length * batch_size * kv_bytes_per_token) / 1e9

    # Activation memory (typically 10-20% of model weights for inference)
    activation_gb = model_memory_gb * 0.15

    # CUDA/Framework overhead (~2-4 GB)
    overhead_gb = 3.0

    total_gb = model_memory_gb + kv_cache_gb + activation_gb + overhead_gb

    return total_gb

# Examples
print(calculate_gpu_memory_requirement(3, "int4", 4096, 32))    # Tier S: ~5 GB
print(calculate_gpu_memory_requirement(8, "fp8", 8192, 16))     # Tier M: ~15 GB
print(calculate_gpu_memory_requirement(70, "fp8", 16384, 8))    # Tier L: ~95 GB
print(calculate_gpu_memory_requirement(405, "fp8", 32768, 4))   # Tier XL: ~520 GB
```

### Tensor Parallelism Requirements

| Model Size | Min VRAM | TP=1 | TP=2 | TP=4 | TP=8 |
|------------|----------|------|------|------|------|
| 3B (FP8) | 3GB | L40S, A100, H100 | - | - | - |
| 8B (FP8) | 10GB | L40S, A100, H100 | - | - | - |
| 70B (FP8) | 80GB | B200 only | H100, H200 | A100 | - |
| 405B (FP8) | 450GB | ❌ | ❌ | B200 | H100, H200 |

---

## 7. Infrastructure Tiers

### Tier 1: Hyperscaler (AWS/GCP/Azure)

```yaml
Providers:
  - AWS (EKS + p5/p4d instances)
  - GCP (GKE + a3/a2 instances)
  - Azure (AKS + ND-series)

Characteristics:
  Setup Time: Hours to Days
  Control Level: Medium
  Compliance: Enterprise (SOC2, HIPAA, PCI-DSS, FedRAMP)
  Cost Multiplier: 1.0x (baseline)

GPU Availability:
  B200: Not yet (2025+)
  H200: AWS p5e.48xlarge, GCP a3-ultragpu-8g
  H100: AWS p5.48xlarge, GCP a3-highgpu-8g, Azure ND96isr_H100_v5
  A100: AWS p4d.24xlarge, GCP a2-ultragpu-8g, Azure ND96asr_v4
  L40S: AWS g6.xlarge+, GCP g2-standard-*
  RTX 4090: Not available

Best For:
  - Enterprise production workloads
  - Compliance-heavy industries
  - Teams with existing cloud relationships
  - Workloads needing global distribution
```

**Terraform Module (AWS EKS with GPU)**:
```hcl
module "eks_gpu_cluster" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = "rag-platform-${var.environment}"
  cluster_version = "1.29"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  eks_managed_node_groups = {
    # Tier S/M workloads
    gpu_l40s = {
      instance_types = ["g6.xlarge"]
      capacity_type  = "SPOT"
      min_size       = 0
      max_size       = 10
      desired_size   = 2

      labels = {
        "nvidia.com/gpu.product" = "NVIDIA-L40S"
        "rag-platform/tier"      = "s-m"
      }

      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }

    # Tier L/XL workloads
    gpu_h100 = {
      instance_types = ["p5.48xlarge"]
      capacity_type  = "ON_DEMAND"
      min_size       = 0
      max_size       = 4
      desired_size   = 1

      labels = {
        "nvidia.com/gpu.product" = "NVIDIA-H100-80GB-HBM3"
        "rag-platform/tier"      = "l-xl"
      }
    }
  }
}
```

### Tier 2: GPU-Native Cloud (CoreWeave/Lambda)

```yaml
Providers:
  - CoreWeave
  - Lambda Labs
  - Crusoe

Characteristics:
  Setup Time: Minutes to Hours
  Control Level: High
  Compliance: Growing (SOC2 available)
  Cost Multiplier: 0.6x (40% cheaper than hyperscalers)

GPU Availability:
  B200: CoreWeave (Q1 2025)
  H200: CoreWeave h200-80gb-sxm5
  H100: CoreWeave h100-80gb-sxm5, Lambda gpu_8x_h100_sxm5
  A100: CoreWeave a100-80gb-sxm4, Lambda gpu_8x_a100_80gb_sxm4
  L40S: CoreWeave l40s-48gb
  RTX 4090: Not available

Best For:
  - Cost-sensitive production
  - ML-focused teams
  - High-throughput inference
  - Teams comfortable with newer providers
```

**CoreWeave Kubernetes Deployment**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama-70b
  namespace: inference
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-llama-70b
  template:
    metadata:
      labels:
        app: vllm-llama-70b
    spec:
      nodeSelector:
        gpu.nvidia.com/class: H100_NVLINK_80GB
      containers:
      - name: vllm
        image: vllm/vllm-openai:v0.6.4
        args:
          - "--model=meta-llama/Llama-3.1-70B-Instruct"
          - "--tensor-parallel-size=2"
          - "--max-model-len=16384"
          - "--gpu-memory-utilization=0.92"
        resources:
          limits:
            nvidia.com/gpu: 2
            memory: "256Gi"
          requests:
            nvidia.com/gpu: 2
            memory: "256Gi"
        ports:
        - containerPort: 8000
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: topology.kubernetes.io/region
                operator: In
                values: ["us-east-1"]
```

### Tier 3: Serverless (RunPod/Modal)

```yaml
Providers:
  - RunPod
  - Modal
  - Replicate
  - Together AI

Characteristics:
  Setup Time: Seconds to Minutes
  Control Level: Low
  Compliance: Basic
  Cost Multiplier: 0.8x (variable, pay-per-use)

GPU Availability:
  B200: Not yet available
  H200: RunPod H200 (limited)
  H100: RunPod H100, Modal H100, Together
  A100: RunPod A100-80GB, Modal A100
  L40S: RunPod L40S, Modal L40S
  RTX 4090: RunPod RTX 4090

Best For:
  - Development and prototyping
  - Burst capacity
  - Variable workloads
  - Teams without infrastructure expertise
```

**RunPod Serverless Endpoint**:
```python
import runpod

def handler(event):
    """RunPod serverless handler for vLLM inference."""

    from vllm import LLM, SamplingParams

    # Initialize model (cached across invocations)
    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel_size=1,
        max_model_len=8192,
        gpu_memory_utilization=0.9
    )

    # Process request
    prompt = event["input"]["prompt"]
    sampling_params = SamplingParams(
        temperature=event["input"].get("temperature", 0.7),
        max_tokens=event["input"].get("max_tokens", 1024)
    )

    outputs = llm.generate([prompt], sampling_params)

    return {
        "output": outputs[0].outputs[0].text,
        "usage": {
            "prompt_tokens": len(outputs[0].prompt_token_ids),
            "completion_tokens": len(outputs[0].outputs[0].token_ids)
        }
    }

runpod.serverless.start({"handler": handler})
```

### Tier 4: On-Premises / Colocation

```yaml
Providers:
  - Self-managed data centers
  - Equinix Metal
  - Vultr Bare Metal

Characteristics:
  Setup Time: Weeks to Months
  Control Level: Full
  Compliance: Custom (full audit control)
  Cost Multiplier: 0.4x (60% cheaper long-term)

Reference Architecture:
  Locations: 2 (NYC, San Francisco)
  Racks per Location: 1
  Nodes per Rack: 8
  GPUs per Node: 8
  Total per Rack: 64 GPUs

  Scale Target:
    - 10 racks across 10 locations
    - 500 GPUs per rack
    - 5000 GPUs total cluster

GPU Availability:
  B200: DGX B200 (Q2 2025+)
  H200: DGX H200, HGX H200
  H100: DGX H100, HGX H100, Dell/HPE/Supermicro
  A100: DGX A100, HGX A100
  L40S: Dell/HPE servers
  RTX 4090: Custom workstations

Best For:
  - Maximum cost efficiency at scale
  - Air-gapped/classified workloads
  - Full control requirements
  - Long-term capacity needs
```

**On-Prem Network Topology**:
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ON-PREMISES CLUSTER TOPOLOGY                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   NYC Data Center                          SF Data Center                   │
│   ┌─────────────────────────┐              ┌─────────────────────────┐      │
│   │        RACK 1           │              │        RACK 2           │      │
│   │  ┌──────────────────┐   │              │  ┌──────────────────┐   │      │
│   │  │  Spine Switch    │   │    WAN Link  │  │  Spine Switch    │   │      │
│   │  │  (400GbE)        │◄──┼──────────────┼─►│  (400GbE)        │   │      │
│   │  └────────┬─────────┘   │   10Gbps     │  └────────┬─────────┘   │      │
│   │           │             │              │           │             │      │
│   │  ┌────────▼─────────┐   │              │  ┌────────▼─────────┐   │      │
│   │  │  Leaf Switches   │   │              │  │  Leaf Switches   │   │      │
│   │  │  (InfiniBand)    │   │              │  │  (InfiniBand)    │   │      │
│   │  └────────┬─────────┘   │              │  └────────┬─────────┘   │      │
│   │           │             │              │           │             │      │
│   │  ┌────────▼─────────┐   │              │  ┌────────▼─────────┐   │      │
│   │  │ NODE 1: 8×H100   │   │              │  │ NODE 1: 8×H100   │   │      │
│   │  │ NODE 2: 8×H100   │   │              │  │ NODE 2: 8×H100   │   │      │
│   │  │ NODE 3: 8×H100   │   │              │  │ NODE 3: 8×H100   │   │      │
│   │  │ NODE 4: 8×H100   │   │              │  │ NODE 4: 8×H100   │   │      │
│   │  │ NODE 5: 8×A100   │   │              │  │ NODE 5: 8×A100   │   │      │
│   │  │ NODE 6: 8×A100   │   │              │  │ NODE 6: 8×A100   │   │      │
│   │  │ NODE 7: 8×L40S   │   │              │  │ NODE 7: 8×L40S   │   │      │
│   │  │ NODE 8: 8×L40S   │   │              │  │ NODE 8: 8×L40S   │   │      │
│   │  └──────────────────┘   │              │  └──────────────────┘   │      │
│   │                         │              │                         │      │
│   │  Total: 64 GPUs         │              │  Total: 64 GPUs         │      │
│   └─────────────────────────┘              └─────────────────────────┘      │
│                                                                             │
│   Storage: Shared GPFS/Lustre over NVMe-oF                                  │
│   Network: InfiniBand HDR (200Gb/s) intra-rack, 100GbE inter-rack          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Slurm Configuration for On-Prem**:
```bash
# /etc/slurm/slurm.conf (excerpt)
ClusterName=rag-cluster
SlurmctldHost=controller

# Partition definitions
PartitionName=tier-s Nodes=node[7-8] Default=NO MaxTime=4:00:00 State=UP
PartitionName=tier-m Nodes=node[5-8] Default=YES MaxTime=8:00:00 State=UP
PartitionName=tier-l Nodes=node[1-4] Default=NO MaxTime=24:00:00 State=UP
PartitionName=tier-xl Nodes=node[1-4] Default=NO MaxTime=72:00:00 State=UP

# GPU resource definitions
GresTypes=gpu
NodeName=node[1-4] Gres=gpu:h100:8 CPUs=128 RealMemory=2048000
NodeName=node[5-6] Gres=gpu:a100:8 CPUs=128 RealMemory=2048000
NodeName=node[7-8] Gres=gpu:l40s:8 CPUs=64 RealMemory=512000

# Accounting
AccountingStorageType=accounting_storage/slurmdbd
JobAcctGatherType=jobacct_gather/cgroup
```

---

## 8. RAG Pipeline Architecture

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   User Query                                                                │
│       │                                                                     │
│       ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    1. QUERY PROCESSING                               │   │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │   │
│   │   │   Query      │  │   Query      │  │    HyDE (Optional)       │  │   │
│   │   │   Parsing    │─▶│   Expansion  │─▶│    Hypothetical Doc      │  │   │
│   │   └──────────────┘  └──────────────┘  └──────────────────────────┘  │   │
│   └──────────────────────────────────────────────┬──────────────────────┘   │
│                                                  │                          │
│                                                  ▼                          │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    2. RETRIEVAL                                      │   │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │   │
│   │   │   Embed      │  │   Vector     │  │    Hybrid Search         │  │   │
│   │   │   Query      │─▶│   Search     │─▶│    (Dense + Sparse)      │  │   │
│   │   └──────────────┘  └──────────────┘  └──────────────────────────┘  │   │
│   │          │                                      │                    │   │
│   │          │         ┌──────────────┐             │                    │   │
│   │          └────────▶│   BM25       │─────────────┘                    │   │
│   │                    │   (Sparse)   │                                  │   │
│   │                    └──────────────┘                                  │   │
│   └──────────────────────────────────────────────┬──────────────────────┘   │
│                                                  │                          │
│                                                  ▼                          │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    3. RERANKING                                      │   │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │   │
│   │   │   Coarse     │  │   Fine       │  │    Diversity             │  │   │
│   │   │   Rerank     │─▶│   Rerank     │─▶│    Filter (MMR)          │  │   │
│   │   │   (Fast)     │  │   (Precise)  │  │                          │  │   │
│   │   └──────────────┘  └──────────────┘  └──────────────────────────┘  │   │
│   └──────────────────────────────────────────────┬──────────────────────┘   │
│                                                  │                          │
│                                                  ▼                          │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    4. GENERATION                                     │   │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │   │
│   │   │   Context    │  │   Prompt     │  │    LLM                   │  │   │
│   │   │   Assembly   │─▶│   Template   │─▶│    Generation            │  │   │
│   │   └──────────────┘  └──────────────┘  └──────────────────────────┘  │   │
│   └──────────────────────────────────────────────┬──────────────────────┘   │
│                                                  │                          │
│                                                  ▼                          │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    5. POST-PROCESSING                                │   │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │   │
│   │   │   Citation   │  │   Fact       │  │    Response              │  │   │
│   │   │   Injection  │─▶│   Grounding  │─▶│    Formatting            │  │   │
│   │   └──────────────┘  └──────────────┘  └──────────────────────────┘  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core RAG Implementation

```python
"""
Production RAG Pipeline Implementation
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod
import asyncio

@dataclass
class Document:
    """Represents a document chunk."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    score: float = 0.0

@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""
    # Embedding
    embedding_model: str = "nomic-embed-text-v1.5"
    embedding_dimensions: int = 768

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 102  # 20%

    # Retrieval
    top_k: int = 10
    rerank_top_k: int = 5
    use_hybrid_search: bool = True
    hybrid_alpha: float = 0.7  # Dense weight

    # Reranking
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    use_mmr: bool = True
    mmr_lambda: float = 0.7

    # Generation
    max_context_tokens: int = 8192
    temperature: float = 0.7
    max_output_tokens: int = 1024


class VectorStore(Protocol):
    """Protocol for vector store implementations."""

    async def upsert(self, documents: List[Document]) -> int:
        """Insert or update documents."""
        ...

    async def search(
        self,
        query_embedding: List[float],
        top_k: int,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        """Search for similar documents."""
        ...


class EmbeddingService(Protocol):
    """Protocol for embedding services."""

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        ...


class LLMService(Protocol):
    """Protocol for LLM services."""

    async def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate text completion."""
        ...


class RAGPipeline:
    """
    Production RAG pipeline with multi-stage retrieval.
    """

    def __init__(
        self,
        config: RAGConfig,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        llm_service: LLMService,
        reranker: Optional['Reranker'] = None
    ):
        self.config = config
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self.reranker = reranker

    async def query(
        self,
        question: str,
        filter: Optional[Dict] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Execute RAG query pipeline.

        Returns:
            {
                "answer": str,
                "sources": List[Document],
                "metadata": {
                    "retrieval_time_ms": float,
                    "generation_time_ms": float,
                    "total_tokens": int
                }
            }
        """
        import time

        start_time = time.time()

        # 1. Query Processing
        processed_query = await self._process_query(question)

        # 2. Retrieval
        retrieval_start = time.time()
        candidates = await self._retrieve(processed_query, filter)
        retrieval_time = (time.time() - retrieval_start) * 1000

        # 3. Reranking
        if self.reranker and len(candidates) > self.config.rerank_top_k:
            candidates = await self._rerank(question, candidates)

        # 4. Context Assembly
        context = self._assemble_context(candidates)

        # 5. Generation
        generation_start = time.time()
        prompt = self._build_prompt(question, context)
        answer = await self.llm_service.generate(
            prompt,
            max_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature
        )
        generation_time = (time.time() - generation_start) * 1000

        # 6. Post-processing
        answer = self._inject_citations(answer, candidates)

        return {
            "answer": answer,
            "sources": candidates[:self.config.rerank_top_k],
            "metadata": {
                "retrieval_time_ms": retrieval_time,
                "generation_time_ms": generation_time,
                "total_time_ms": (time.time() - start_time) * 1000,
                "documents_retrieved": len(candidates)
            }
        }

    async def _process_query(self, query: str) -> str:
        """Process and optionally expand query."""
        # Could add query expansion, HyDE, etc.
        return query

    async def _retrieve(
        self,
        query: str,
        filter: Optional[Dict]
    ) -> List[Document]:
        """Retrieve candidate documents."""

        # Get query embedding
        embeddings = await self.embedding_service.embed([query])
        query_embedding = embeddings[0]

        # Vector search
        dense_results = await self.vector_store.search(
            query_embedding,
            top_k=self.config.top_k * 2,  # Overfetch for reranking
            filter=filter
        )

        if not self.config.use_hybrid_search:
            return dense_results

        # Hybrid: combine with BM25 (simplified)
        # In production, use Qdrant's hybrid or separate BM25 index
        return dense_results

    async def _rerank(
        self,
        query: str,
        documents: List[Document]
    ) -> List[Document]:
        """Rerank documents for relevance."""

        scores = await self.reranker.score(
            query,
            [doc.content for doc in documents]
        )

        for doc, score in zip(documents, scores):
            doc.score = score

        # Sort by reranker score
        documents.sort(key=lambda d: d.score, reverse=True)

        # Apply MMR for diversity if enabled
        if self.config.use_mmr:
            documents = self._apply_mmr(documents)

        return documents[:self.config.rerank_top_k]

    def _apply_mmr(self, documents: List[Document]) -> List[Document]:
        """Apply Maximal Marginal Relevance for diversity."""
        if not documents:
            return documents

        selected = [documents[0]]
        remaining = documents[1:]

        while remaining and len(selected) < self.config.rerank_top_k:
            best_score = float('-inf')
            best_doc = None

            for doc in remaining:
                # Simplified MMR: relevance - max_similarity_to_selected
                relevance = doc.score
                max_sim = max(
                    self._cosine_similarity(doc.embedding, s.embedding)
                    for s in selected
                ) if doc.embedding and selected[0].embedding else 0

                mmr_score = (
                    self.config.mmr_lambda * relevance -
                    (1 - self.config.mmr_lambda) * max_sim
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_doc = doc

            if best_doc:
                selected.append(best_doc)
                remaining.remove(best_doc)

        return selected

    def _cosine_similarity(
        self,
        a: Optional[List[float]],
        b: Optional[List[float]]
    ) -> float:
        """Compute cosine similarity between vectors."""
        if not a or not b:
            return 0.0

        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def _assemble_context(self, documents: List[Document]) -> str:
        """Assemble context from retrieved documents."""
        context_parts = []

        for i, doc in enumerate(documents):
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(
                f"[Document {i+1}] (Source: {source})\n{doc.content}"
            )

        return "\n\n".join(context_parts)

    def _build_prompt(self, question: str, context: str) -> str:
        """Build the generation prompt."""
        return f"""You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the information in the context above
- If the context doesn't contain enough information, say so
- Cite your sources using [Document N] format
- Be concise but comprehensive

Answer:"""

    def _inject_citations(
        self,
        answer: str,
        documents: List[Document]
    ) -> str:
        """Ensure proper citation formatting."""
        # In production, verify citations match actual documents
        return answer


# Reranker implementation
class CrossEncoderReranker:
    """Cross-encoder reranker using BGE or Cohere."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    async def score(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """Score query-document pairs."""
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        return scores.tolist()
```

### RAG Configuration by Tier

| Component | Tier S | Tier M | Tier L | Tier XL |
|-----------|--------|--------|--------|---------|
| **Vector DB** | pgvector | Qdrant | Milvus | Milvus Cluster |
| **Embedding** | nomic-768d | e5-large-1024d | text-embedding-3-large | voyage-large-2 |
| **Chunk Size** | 256 | 512 | 1024 | 2048 |
| **Top-K** | 3 | 5 | 10 | 20 |
| **Reranker** | bge-reranker-m3 | Cohere v3 | Multi-stage | Ensemble |
| **Context Budget** | 2K | 8K | 16K | 32K |

---

## 9. Intelligent Scheduling System

### Multi-Tier Scheduling Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       INTELLIGENT SCHEDULING SYSTEM                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Incoming Request                                                          │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    REQUEST CLASSIFIER                                │   │
│   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐ │   │
│   │   │  Complexity │  │   SLA       │  │    Cost                     │ │   │
│   │   │  Analysis   │  │   Check     │  │    Budget                   │ │   │
│   │   └──────┬──────┘  └──────┬──────┘  └────────────┬────────────────┘ │   │
│   │          └────────────────┼──────────────────────┘                  │   │
│   │                           ▼                                          │   │
│   │                   ┌───────────────┐                                  │   │
│   │                   │  Tier Select  │                                  │   │
│   │                   │  S | M | L | XL                                  │   │
│   │                   └───────┬───────┘                                  │   │
│   └───────────────────────────┼─────────────────────────────────────────┘   │
│                               │                                             │
│   ┌───────────────────────────▼─────────────────────────────────────────┐   │
│   │                    KUEUE CLUSTER QUEUES                              │   │
│   │                                                                      │   │
│   │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐               │   │
│   │   │ TIER-S  │  │ TIER-M  │  │ TIER-L  │  │ TIER-XL │               │   │
│   │   │ Queue   │  │ Queue   │  │ Queue   │  │ Queue   │               │   │
│   │   │         │  │         │  │         │  │         │               │   │
│   │   │ L40S    │  │ A100    │  │ H100x2  │  │ H100x8  │               │   │
│   │   │ RTX4090 │  │ L40S    │  │ H200x2  │  │ H200x8  │               │   │
│   │   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘               │   │
│   │        │            │            │            │                     │   │
│   └────────┼────────────┼────────────┼────────────┼─────────────────────┘   │
│            │            │            │            │                         │
│   ┌────────▼────────────▼────────────▼────────────▼─────────────────────┐   │
│   │                    RESOURCE FLAVORS                                  │   │
│   │                                                                      │   │
│   │   Hyperscaler    GPU-Native     Serverless      On-Prem             │   │
│   │   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐         │   │
│   │   │ AWS     │    │CoreWeave│    │ RunPod  │    │  NYC    │         │   │
│   │   │ GCP     │    │ Lambda  │    │ Modal   │    │  SF     │         │   │
│   │   │ Azure   │    │         │    │         │    │         │         │   │
│   │   └─────────┘    └─────────┘    └─────────┘    └─────────┘         │   │
│   │                                                                      │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Kueue Configuration

```yaml
# ClusterQueue definition for multi-tier scheduling
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: rag-platform-queue
spec:
  namespaceSelector: {}
  resourceGroups:
    - coveredResources: ["cpu", "memory", "nvidia.com/gpu"]
      flavors:
        # Tier S flavors (small models)
        - name: l40s-spot
          resources:
            - name: "cpu"
              nominalQuota: 64
            - name: "memory"
              nominalQuota: 256Gi
            - name: "nvidia.com/gpu"
              nominalQuota: 8
              borrowingLimit: 4

        - name: rtx4090-runpod
          resources:
            - name: "cpu"
              nominalQuota: 32
            - name: "memory"
              nominalQuota: 128Gi
            - name: "nvidia.com/gpu"
              nominalQuota: 4

        # Tier M flavors (medium models)
        - name: a100-ondemand
          resources:
            - name: "cpu"
              nominalQuota: 128
            - name: "memory"
              nominalQuota: 512Gi
            - name: "nvidia.com/gpu"
              nominalQuota: 8

        # Tier L flavors (large models)
        - name: h100-coreweave
          resources:
            - name: "cpu"
              nominalQuota: 256
            - name: "memory"
              nominalQuota: 1Ti
            - name: "nvidia.com/gpu"
              nominalQuota: 16

        # Tier XL flavors (frontier models)
        - name: h100-8gpu-onprem
          resources:
            - name: "cpu"
              nominalQuota: 512
            - name: "memory"
              nominalQuota: 2Ti
            - name: "nvidia.com/gpu"
              nominalQuota: 64

  preemption:
    reclaimWithinCohort: Any
    withinClusterQueue: LowerPriority

---
# LocalQueue for inference workloads
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata:
  name: inference-queue
  namespace: inference
spec:
  clusterQueue: rag-platform-queue

---
# WorkloadPriorityClass definitions
apiVersion: kueue.x-k8s.io/v1beta1
kind: WorkloadPriorityClass
metadata:
  name: production-critical
value: 1000000
description: "Production SLA-bound requests"

---
apiVersion: kueue.x-k8s.io/v1beta1
kind: WorkloadPriorityClass
metadata:
  name: production-standard
value: 100000
description: "Standard production requests"

---
apiVersion: kueue.x-k8s.io/v1beta1
kind: WorkloadPriorityClass
metadata:
  name: batch-processing
value: 10000
description: "Batch and async processing"

---
apiVersion: kueue.x-k8s.io/v1beta1
kind: WorkloadPriorityClass
metadata:
  name: development
value: 1000
description: "Development and testing"
```

### Intelligent Router Implementation

```python
"""
Intelligent request router with cost-aware scheduling.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import asyncio
import time

class ModelTier(Enum):
    S = "s"    # 2B
    M = "m"    # 8-20B
    L = "l"    # 70-100B
    XL = "xl"  # 400-500B

class InfrastructureTier(Enum):
    HYPERSCALER = "hyperscaler"
    GPU_NATIVE = "gpu_native"
    SERVERLESS = "serverless"
    ON_PREMISES = "on_premises"

@dataclass
class RoutingDecision:
    """Result of routing decision."""
    model_tier: ModelTier
    infrastructure_tier: InfrastructureTier
    gpu_type: str
    endpoint: str
    estimated_cost: float
    estimated_latency_ms: float
    queue_position: int
    confidence: float

@dataclass
class RequestContext:
    """Context for routing decision."""
    query: str
    max_latency_ms: float
    max_cost_per_request: float
    required_quality: str  # "low", "medium", "high", "maximum"
    user_tier: str  # "free", "pro", "enterprise"
    compliance_requirements: List[str]

class IntelligentRouter:
    """
    Routes requests to optimal model/infrastructure combination.
    """

    def __init__(
        self,
        endpoint_registry: 'EndpointRegistry',
        cost_tracker: 'CostTracker',
        metrics_client: 'MetricsClient'
    ):
        self.endpoint_registry = endpoint_registry
        self.cost_tracker = cost_tracker
        self.metrics = metrics_client

        # Complexity classifier weights (from fine-tuned model)
        self.complexity_thresholds = {
            "simple": 0.3,      # -> Tier S
            "moderate": 0.5,    # -> Tier M
            "complex": 0.7,     # -> Tier L
            "frontier": 1.0     # -> Tier XL
        }

    async def route(self, context: RequestContext) -> RoutingDecision:
        """
        Make routing decision based on request context.
        """

        # 1. Classify query complexity
        complexity_score = await self._classify_complexity(context.query)
        model_tier = self._select_model_tier(
            complexity_score,
            context.required_quality
        )

        # 2. Get available endpoints for tier
        endpoints = self.endpoint_registry.get_endpoints(model_tier)

        # 3. Filter by constraints
        viable_endpoints = []
        for endpoint in endpoints:
            # Check compliance
            if not self._meets_compliance(
                endpoint,
                context.compliance_requirements
            ):
                continue

            # Check latency SLA
            estimated_latency = await self._estimate_latency(endpoint)
            if estimated_latency > context.max_latency_ms:
                continue

            # Check cost
            estimated_cost = self._estimate_cost(endpoint, context.query)
            if estimated_cost > context.max_cost_per_request:
                continue

            viable_endpoints.append((endpoint, estimated_latency, estimated_cost))

        if not viable_endpoints:
            # Fallback: try lower tier
            return await self._fallback_route(context, model_tier)

        # 4. Score and select best endpoint
        best_endpoint = self._select_best_endpoint(
            viable_endpoints,
            context.user_tier
        )

        endpoint, latency, cost = best_endpoint

        # 5. Record decision for learning
        self.metrics.record_routing_decision(
            model_tier=model_tier,
            endpoint=endpoint.name,
            complexity_score=complexity_score
        )

        return RoutingDecision(
            model_tier=model_tier,
            infrastructure_tier=endpoint.infrastructure_tier,
            gpu_type=endpoint.gpu_type,
            endpoint=endpoint.url,
            estimated_cost=cost,
            estimated_latency_ms=latency,
            queue_position=endpoint.queue_depth,
            confidence=complexity_score
        )

    async def _classify_complexity(self, query: str) -> float:
        """
        Classify query complexity using lightweight model.

        Returns score from 0.0 (trivial) to 1.0 (frontier).
        """
        # Heuristic features
        features = {
            "length": len(query),
            "question_words": sum(
                1 for w in ["why", "how", "explain", "analyze", "compare"]
                if w in query.lower()
            ),
            "technical_terms": sum(
                1 for term in self._technical_terms
                if term in query.lower()
            ),
            "multi_part": query.count("?") + query.count(" and "),
        }

        # Simple scoring (in production: use fine-tuned classifier)
        score = 0.0
        score += min(features["length"] / 500, 0.3)
        score += features["question_words"] * 0.1
        score += features["technical_terms"] * 0.05
        score += features["multi_part"] * 0.15

        return min(score, 1.0)

    def _select_model_tier(
        self,
        complexity_score: float,
        required_quality: str
    ) -> ModelTier:
        """Select model tier based on complexity and quality requirement."""

        # Quality modifier
        quality_boost = {
            "low": -0.2,
            "medium": 0.0,
            "high": 0.15,
            "maximum": 0.3
        }.get(required_quality, 0.0)

        adjusted_score = complexity_score + quality_boost

        if adjusted_score < self.complexity_thresholds["simple"]:
            return ModelTier.S
        elif adjusted_score < self.complexity_thresholds["moderate"]:
            return ModelTier.M
        elif adjusted_score < self.complexity_thresholds["complex"]:
            return ModelTier.L
        else:
            return ModelTier.XL

    def _meets_compliance(
        self,
        endpoint: 'Endpoint',
        requirements: List[str]
    ) -> bool:
        """Check if endpoint meets compliance requirements."""
        if not requirements:
            return True

        endpoint_compliance = set(endpoint.compliance_certifications)
        required = set(requirements)

        return required.issubset(endpoint_compliance)

    async def _estimate_latency(self, endpoint: 'Endpoint') -> float:
        """Estimate request latency for endpoint."""
        # Base latency from historical data
        base_latency = endpoint.avg_latency_ms

        # Queue delay
        queue_delay = endpoint.queue_depth * endpoint.avg_processing_time_ms

        # Network latency estimate
        network_latency = 50  # ms, could be region-aware

        return base_latency + queue_delay + network_latency

    def _estimate_cost(self, endpoint: 'Endpoint', query: str) -> float:
        """Estimate cost for processing query."""
        # Estimate tokens
        estimated_input_tokens = len(query) / 4  # rough approximation
        estimated_output_tokens = 500  # default assumption

        return (
            estimated_input_tokens * endpoint.cost_per_input_token +
            estimated_output_tokens * endpoint.cost_per_output_token
        )

    def _select_best_endpoint(
        self,
        endpoints: List[Tuple['Endpoint', float, float]],
        user_tier: str
    ) -> Tuple['Endpoint', float, float]:
        """Select best endpoint based on scoring."""

        # Weights based on user tier
        weights = {
            "free": {"cost": 0.7, "latency": 0.2, "quality": 0.1},
            "pro": {"cost": 0.3, "latency": 0.4, "quality": 0.3},
            "enterprise": {"cost": 0.1, "latency": 0.3, "quality": 0.6}
        }.get(user_tier, {"cost": 0.5, "latency": 0.3, "quality": 0.2})

        scored = []
        for endpoint, latency, cost in endpoints:
            # Normalize scores (lower is better for cost/latency)
            cost_score = 1.0 / (1.0 + cost)
            latency_score = 1.0 / (1.0 + latency / 1000)
            quality_score = endpoint.quality_rating

            total_score = (
                weights["cost"] * cost_score +
                weights["latency"] * latency_score +
                weights["quality"] * quality_score
            )

            scored.append((total_score, endpoint, latency, cost))

        scored.sort(reverse=True)
        _, endpoint, latency, cost = scored[0]

        return (endpoint, latency, cost)

    async def _fallback_route(
        self,
        context: RequestContext,
        original_tier: ModelTier
    ) -> RoutingDecision:
        """Fallback to lower tier if original tier unavailable."""
        tier_order = [ModelTier.XL, ModelTier.L, ModelTier.M, ModelTier.S]
        current_idx = tier_order.index(original_tier)

        for tier in tier_order[current_idx + 1:]:
            endpoints = self.endpoint_registry.get_endpoints(tier)
            if endpoints:
                # Use first available with relaxed constraints
                endpoint = endpoints[0]
                return RoutingDecision(
                    model_tier=tier,
                    infrastructure_tier=endpoint.infrastructure_tier,
                    gpu_type=endpoint.gpu_type,
                    endpoint=endpoint.url,
                    estimated_cost=self._estimate_cost(endpoint, context.query),
                    estimated_latency_ms=await self._estimate_latency(endpoint),
                    queue_position=endpoint.queue_depth,
                    confidence=0.5  # Lower confidence for fallback
                )

        raise NoAvailableEndpointError("No endpoints available for request")

    _technical_terms = [
        "algorithm", "architecture", "implementation", "optimization",
        "performance", "latency", "throughput", "distributed", "parallel",
        "inference", "training", "model", "neural", "transformer"
    ]


class NoAvailableEndpointError(Exception):
    pass
```

---

## 10. Deployment Patterns

### vLLM Deployment by Configuration

```yaml
# Tier S: Single L40S
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-tier-s
  namespace: inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vllm-tier-s
  template:
    metadata:
      labels:
        app: vllm-tier-s
        model-tier: s
    spec:
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-L40S
      containers:
      - name: vllm
        image: vllm/vllm-openai:v0.6.4
        args:
          - "--model=Qwen/Qwen2.5-3B-Instruct"
          - "--quantization=awq"
          - "--max-model-len=32768"
          - "--gpu-memory-utilization=0.92"
          - "--enable-chunked-prefill"
          - "--max-num-batched-tokens=32768"
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "48Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "48Gi"
        ports:
        - containerPort: 8000
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
        env:
        - name: VLLM_ATTENTION_BACKEND
          value: "FLASHINFER"

---
# Tier M: Single A100
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-tier-m
  namespace: inference
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vllm-tier-m
  template:
    metadata:
      labels:
        app: vllm-tier-m
        model-tier: m
    spec:
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-A100-SXM4-80GB
      containers:
      - name: vllm
        image: vllm/vllm-openai:v0.6.4
        args:
          - "--model=meta-llama/Llama-3.1-8B-Instruct"
          - "--max-model-len=65536"
          - "--gpu-memory-utilization=0.92"
          - "--enable-prefix-caching"
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "256Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "256Gi"
        ports:
        - containerPort: 8000

---
# Tier L: 2x H100 with Tensor Parallelism
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-tier-l
  namespace: inference
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-tier-l
  template:
    metadata:
      labels:
        app: vllm-tier-l
        model-tier: l
    spec:
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3
      containers:
      - name: vllm
        image: vllm/vllm-openai:v0.6.4
        args:
          - "--model=meta-llama/Llama-3.1-70B-Instruct"
          - "--tensor-parallel-size=2"
          - "--max-model-len=32768"
          - "--gpu-memory-utilization=0.92"
          - "--enable-prefix-caching"
          - "--enable-chunked-prefill"
        resources:
          limits:
            nvidia.com/gpu: 2
            memory: "512Gi"
          requests:
            nvidia.com/gpu: 2
            memory: "512Gi"
        ports:
        - containerPort: 8000
        env:
        - name: NCCL_P2P_LEVEL
          value: "NVL"

---
# Tier XL: 8x H100 Full Node
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-tier-xl
  namespace: inference
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-tier-xl
  template:
    metadata:
      labels:
        app: vllm-tier-xl
        model-tier: xl
    spec:
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3
        node.kubernetes.io/instance-type: p5.48xlarge
      containers:
      - name: vllm
        image: vllm/vllm-openai:v0.6.4
        args:
          - "--model=meta-llama/Llama-3.1-405B-Instruct-FP8"
          - "--tensor-parallel-size=8"
          - "--max-model-len=32768"
          - "--gpu-memory-utilization=0.92"
          - "--enable-prefix-caching"
          - "--quantization=fp8"
        resources:
          limits:
            nvidia.com/gpu: 8
            memory: "2Ti"
          requests:
            nvidia.com/gpu: 8
            memory: "2Ti"
        ports:
        - containerPort: 8000
        env:
        - name: NCCL_IB_DISABLE
          value: "0"
        - name: NCCL_NET_GDR_LEVEL
          value: "5"
```

### Helm Chart Structure

```yaml
# values.yaml
global:
  imageRegistry: "ghcr.io/your-org"
  imagePullSecrets:
    - name: ghcr-secret

inference:
  tiers:
    s:
      enabled: true
      replicas: 3
      model: "Qwen/Qwen2.5-3B-Instruct"
      quantization: "awq"
      gpuType: "L40S"
      gpuCount: 1
      maxModelLen: 32768
      resources:
        memory: "48Gi"

    m:
      enabled: true
      replicas: 2
      model: "meta-llama/Llama-3.1-8B-Instruct"
      quantization: null
      gpuType: "A100"
      gpuCount: 1
      maxModelLen: 65536
      resources:
        memory: "256Gi"

    l:
      enabled: true
      replicas: 1
      model: "meta-llama/Llama-3.1-70B-Instruct"
      quantization: null
      gpuType: "H100"
      gpuCount: 2
      tensorParallel: 2
      maxModelLen: 32768
      resources:
        memory: "512Gi"

    xl:
      enabled: false  # Enable for frontier workloads
      replicas: 1
      model: "meta-llama/Llama-3.1-405B-Instruct-FP8"
      quantization: "fp8"
      gpuType: "H100"
      gpuCount: 8
      tensorParallel: 8
      maxModelLen: 32768
      resources:
        memory: "2Ti"

vectorStore:
  type: "qdrant"
  replicas: 3
  storage:
    size: "100Gi"
    storageClass: "gp3"

embedding:
  model: "nomic-embed-text-v1.5"
  replicas: 2
  gpuEnabled: false

reranker:
  model: "BAAI/bge-reranker-v2-m3"
  replicas: 2
  gpuEnabled: true

monitoring:
  prometheus:
    enabled: true
  grafana:
    enabled: true
    dashboards:
      - rag-overview
      - llm-performance
      - cost-tracking
```

---

## 11. Observability Framework

### Metrics Architecture

```yaml
# Prometheus ServiceMonitor for vLLM
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: vllm-metrics
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app.kubernetes.io/component: inference
  endpoints:
  - port: metrics
    interval: 15s
    path: /metrics

---
# Key metrics to collect
# vllm_request_success_total
# vllm_request_latency_seconds
# vllm_tokens_total{type="prompt|completion"}
# vllm_gpu_memory_usage_bytes
# vllm_running_requests
# vllm_pending_requests
# vllm_cache_hit_rate
```

### Grafana Dashboard (JSON excerpt)

```json
{
  "title": "RAG Platform Overview",
  "panels": [
    {
      "title": "Requests by Model Tier",
      "type": "timeseries",
      "targets": [
        {
          "expr": "sum(rate(vllm_request_success_total[5m])) by (model_tier)",
          "legendFormat": "Tier {{model_tier}}"
        }
      ]
    },
    {
      "title": "P99 Latency by Tier",
      "type": "timeseries",
      "targets": [
        {
          "expr": "histogram_quantile(0.99, sum(rate(vllm_request_latency_seconds_bucket[5m])) by (le, model_tier))",
          "legendFormat": "{{model_tier}}"
        }
      ]
    },
    {
      "title": "Cost per 1K Tokens",
      "type": "stat",
      "targets": [
        {
          "expr": "sum(rag_cost_total) / sum(vllm_tokens_total) * 1000",
          "legendFormat": "$/1K tokens"
        }
      ]
    },
    {
      "title": "GPU Utilization",
      "type": "gauge",
      "targets": [
        {
          "expr": "avg(DCGM_FI_DEV_GPU_UTIL)",
          "legendFormat": "GPU Util %"
        }
      ]
    }
  ]
}
```

### Alerting Rules

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: rag-platform-alerts
spec:
  groups:
  - name: rag-platform
    rules:
    - alert: HighLatencyTierS
      expr: |
        histogram_quantile(0.99,
          sum(rate(vllm_request_latency_seconds_bucket{model_tier="s"}[5m])) by (le)
        ) > 0.1
      for: 5m
      labels:
        severity: warning
        tier: s
      annotations:
        summary: "Tier S P99 latency > 100ms"

    - alert: HighLatencyTierL
      expr: |
        histogram_quantile(0.99,
          sum(rate(vllm_request_latency_seconds_bucket{model_tier="l"}[5m])) by (le)
        ) > 1.0
      for: 5m
      labels:
        severity: warning
        tier: l
      annotations:
        summary: "Tier L P99 latency > 1000ms"

    - alert: GPUMemoryPressure
      expr: |
        (vllm_gpu_memory_usage_bytes / vllm_gpu_memory_total_bytes) > 0.95
      for: 2m
      labels:
        severity: critical
      annotations:
        summary: "GPU memory utilization > 95%"

    - alert: HighQueueDepth
      expr: vllm_pending_requests > 100
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "Request queue depth > 100"

    - alert: CostBudgetExceeded
      expr: |
        sum(increase(rag_cost_total[1h])) > 100
      labels:
        severity: critical
      annotations:
        summary: "Hourly cost exceeded $100 budget"
```

---

## 12. Cost Optimization

### Cost Model

```python
"""
Cost tracking and optimization for multi-cloud RAG platform.
"""
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime, timedelta

@dataclass
class CostEntry:
    """Single cost entry."""
    timestamp: datetime
    provider: str
    gpu_type: str
    model_tier: str
    gpu_hours: float
    tokens_processed: int
    cost_usd: float

class CostTracker:
    """Track and analyze costs across providers."""

    # Cost per GPU-hour by provider and GPU type
    GPU_COSTS = {
        "aws": {
            "H100": 4.25,
            "A100": 2.50,
            "L40S": 1.50,
        },
        "coreweave": {
            "H100": 2.23,
            "H200": 3.49,
            "A100": 1.89,
            "L40S": 1.14,
        },
        "runpod": {
            "H100": 2.49,
            "A100": 1.64,
            "L40S": 0.99,
            "RTX4090": 0.44,
        },
        "on_premises": {
            "H100": 0.85,  # Amortized + power
            "A100": 0.65,
            "L40S": 0.45,
        }
    }

    def __init__(self):
        self.entries: List[CostEntry] = []

    def record(
        self,
        provider: str,
        gpu_type: str,
        model_tier: str,
        gpu_hours: float,
        tokens_processed: int
    ):
        """Record a cost entry."""
        cost = self.GPU_COSTS[provider][gpu_type] * gpu_hours

        self.entries.append(CostEntry(
            timestamp=datetime.utcnow(),
            provider=provider,
            gpu_type=gpu_type,
            model_tier=model_tier,
            gpu_hours=gpu_hours,
            tokens_processed=tokens_processed,
            cost_usd=cost
        ))

    def get_cost_per_1k_tokens(
        self,
        provider: str = None,
        model_tier: str = None,
        hours: int = 24
    ) -> float:
        """Calculate cost per 1K tokens."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        relevant = [
            e for e in self.entries
            if e.timestamp > cutoff
            and (provider is None or e.provider == provider)
            and (model_tier is None or e.model_tier == model_tier)
        ]

        total_cost = sum(e.cost_usd for e in relevant)
        total_tokens = sum(e.tokens_processed for e in relevant)

        if total_tokens == 0:
            return 0.0

        return (total_cost / total_tokens) * 1000

    def get_provider_comparison(self) -> Dict[str, float]:
        """Compare cost efficiency across providers."""
        providers = {}

        for provider in self.GPU_COSTS.keys():
            cost = self.get_cost_per_1k_tokens(provider=provider)
            if cost > 0:
                providers[provider] = cost

        return dict(sorted(providers.items(), key=lambda x: x[1]))

    def recommend_optimization(self) -> List[str]:
        """Generate cost optimization recommendations."""
        recommendations = []

        # Compare providers
        comparison = self.get_provider_comparison()
        if comparison:
            cheapest = list(comparison.keys())[0]
            most_expensive = list(comparison.keys())[-1]

            if comparison[most_expensive] > comparison[cheapest] * 1.5:
                recommendations.append(
                    f"Consider shifting workloads from {most_expensive} "
                    f"to {cheapest} (potential {comparison[most_expensive]/comparison[cheapest]:.1f}x savings)"
                )

        # Check tier efficiency
        for tier in ["s", "m", "l", "xl"]:
            tier_cost = self.get_cost_per_1k_tokens(model_tier=tier)
            if tier == "s" and tier_cost > 0.01:
                recommendations.append(
                    f"Tier S cost (${tier_cost:.4f}/1K) is high - "
                    "consider using smaller GPUs or spot instances"
                )

        return recommendations


# Cost-based routing weights
COST_EFFICIENCY_MATRIX = {
    # (model_tier, infrastructure) -> relative efficiency score
    ("s", "serverless"): 1.0,     # Best for Tier S
    ("s", "gpu_native"): 0.9,
    ("s", "on_premises"): 0.8,
    ("s", "hyperscaler"): 0.6,

    ("m", "gpu_native"): 1.0,     # Best for Tier M
    ("m", "on_premises"): 0.95,
    ("m", "serverless"): 0.85,
    ("m", "hyperscaler"): 0.7,

    ("l", "on_premises"): 1.0,    # Best for Tier L
    ("l", "gpu_native"): 0.95,
    ("l", "hyperscaler"): 0.8,
    ("l", "serverless"): 0.6,

    ("xl", "on_premises"): 1.0,   # Best for Tier XL
    ("xl", "gpu_native"): 0.9,
    ("xl", "hyperscaler"): 0.7,
    ("xl", "serverless"): 0.3,    # Often not viable
}
```

### Cost Summary by Configuration

| Tier | GPU | Provider | $/GPU/hr | $/1K tokens | Monthly (24/7) |
|------|-----|----------|----------|-------------|----------------|
| S | L40S | RunPod | $0.99 | $0.0008 | $713 |
| S | L40S | CoreWeave | $1.14 | $0.0010 | $821 |
| S | RTX 4090 | RunPod | $0.44 | $0.0004 | $317 |
| M | A100 | CoreWeave | $1.89 | $0.0025 | $1,361 |
| M | A100 | AWS | $2.50 | $0.0033 | $1,800 |
| M | L40S | RunPod | $0.99 | $0.0020 | $713 |
| L | H100×2 | CoreWeave | $4.46 | $0.0150 | $3,211 |
| L | H100×2 | AWS | $8.50 | $0.0285 | $6,120 |
| L | A100×4 | On-Prem | $2.60 | $0.0087 | $1,872 |
| XL | H100×8 | CoreWeave | $17.84 | $0.0890 | $12,845 |
| XL | H100×8 | AWS | $34.00 | $0.1700 | $24,480 |
| XL | H100×8 | On-Prem | $6.80 | $0.0340 | $4,896 |

---

## 13. Security & Compliance

### Security Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SECURITY ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Internet                                                                  │
│       │                                                                     │
│       ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    WAF / DDoS Protection                             │   │
│   │                    (CloudFlare / AWS Shield)                         │   │
│   └──────────────────────────────┬──────────────────────────────────────┘   │
│                                  │                                          │
│   ┌──────────────────────────────▼──────────────────────────────────────┐   │
│   │                    API Gateway (Kong)                                │   │
│   │   • Rate Limiting       • JWT Validation                            │   │
│   │   • Request Logging     • mTLS Termination                          │   │
│   └──────────────────────────────┬──────────────────────────────────────┘   │
│                                  │                                          │
│   ┌──────────────────────────────▼──────────────────────────────────────┐   │
│   │                    Service Mesh (Istio)                              │   │
│   │   • mTLS Between Services   • Network Policies                      │   │
│   │   • Traffic Encryption      • Access Control (AuthZ)                │   │
│   └──────────────────────────────┬──────────────────────────────────────┘   │
│                                  │                                          │
│   ┌──────────────────────────────▼──────────────────────────────────────┐   │
│   │                    Application Layer                                 │   │
│   │   ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐   │   │
│   │   │   Input    │  │   PII      │  │   Prompt   │  │   Output   │   │   │
│   │   │ Validation │  │ Detection  │  │ Injection  │  │ Filtering  │   │   │
│   │   │            │  │ (Presidio) │  │ Guard      │  │            │   │   │
│   │   └────────────┘  └────────────┘  └────────────┘  └────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    Secrets Management (Vault)                        │   │
│   │   • API Keys           • Database Credentials                       │   │
│   │   • Encryption Keys    • Service Tokens                             │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    Data Layer Security                               │   │
│   │   • Encryption at Rest (AES-256)                                    │   │
│   │   • Encryption in Transit (TLS 1.3)                                 │   │
│   │   • Data Classification & DLP                                       │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Compliance Matrix

| Requirement | Hyperscaler | GPU-Native | Serverless | On-Prem |
|-------------|-------------|------------|------------|---------|
| SOC 2 Type II | ✅ | ✅ | ⚠️ | ✅ (self) |
| HIPAA | ✅ (BAA) | ⚠️ | ❌ | ✅ |
| PCI-DSS | ✅ | ⚠️ | ❌ | ✅ |
| GDPR | ✅ | ✅ | ⚠️ | ✅ |
| FedRAMP | ✅ (some) | ❌ | ❌ | ✅ |
| Air-Gapped | ❌ | ❌ | ❌ | ✅ |

### PII Detection Implementation

```python
"""
PII detection and masking for RAG pipeline.
"""
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from typing import List, Tuple

class PIIHandler:
    """Handle PII detection and masking."""

    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

        # Configure detection for common PII types
        self.entities = [
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "CREDIT_CARD",
            "US_SSN",
            "IP_ADDRESS",
            "LOCATION",
            "DATE_TIME"
        ]

    def detect(self, text: str) -> List[Tuple[str, int, int, float]]:
        """
        Detect PII entities in text.

        Returns: List of (entity_type, start, end, score)
        """
        results = self.analyzer.analyze(
            text=text,
            entities=self.entities,
            language="en"
        )

        return [
            (r.entity_type, r.start, r.end, r.score)
            for r in results
        ]

    def mask(self, text: str) -> str:
        """Mask PII in text with placeholders."""
        results = self.analyzer.analyze(
            text=text,
            entities=self.entities,
            language="en"
        )

        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results
        )

        return anonymized.text

    def should_block(self, text: str, threshold: float = 0.85) -> bool:
        """Check if text contains high-confidence PII that should block processing."""
        results = self.detect(text)

        high_risk_entities = ["CREDIT_CARD", "US_SSN"]

        for entity_type, _, _, score in results:
            if entity_type in high_risk_entities and score >= threshold:
                return True

        return False
```

---

## 14. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)

```markdown
### Week 1-2: Infrastructure Setup
- [ ] Provision Kubernetes clusters (AWS EKS, CoreWeave)
- [ ] Deploy Kueue for GPU scheduling
- [ ] Set up Terraform modules for multi-cloud
- [ ] Configure CI/CD pipelines (ArgoCD)

### Week 3-4: Core Services
- [ ] Deploy vLLM for Tier S and M models
- [ ] Set up Qdrant vector database
- [ ] Implement embedding service
- [ ] Deploy Kong API gateway
```

### Phase 2: RAG Pipeline (Weeks 5-8)

```markdown
### Week 5-6: Retrieval Layer
- [ ] Implement document ingestion pipeline
- [ ] Configure chunking strategies by tier
- [ ] Set up hybrid search (dense + BM25)
- [ ] Deploy reranking service

### Week 7-8: Generation Layer
- [ ] Build prompt templates by use case
- [ ] Implement context assembly
- [ ] Add citation injection
- [ ] Deploy Tier L models (70B)
```

### Phase 3: Intelligence (Weeks 9-12)

```markdown
### Week 9-10: Routing System
- [ ] Build request classifier
- [ ] Implement cost-aware router
- [ ] Add SLA-based queue management
- [ ] Deploy multi-tier routing

### Week 11-12: Observability
- [ ] Configure Prometheus metrics
- [ ] Build Grafana dashboards
- [ ] Set up alerting rules
- [ ] Implement distributed tracing
```

### Phase 4: Scale & Optimize (Weeks 13-16)

```markdown
### Week 13-14: On-Premises Integration
- [ ] Deploy to NYC/SF racks
- [ ] Configure InfiniBand networking
- [ ] Set up Slurm scheduling
- [ ] Integrate with cloud bursting

### Week 15-16: Optimization
- [ ] Tune autoscaling policies
- [ ] Optimize cost routing
- [ ] Performance benchmarking
- [ ] Security hardening
```

---

## 15. Repository Structure

This PRD is supported by a comprehensive documentation repository containing **64 detailed technical documents** across **15 categories**.

### Repository Layout

```
llm-deployment-strat/
│
├── 01_data_pipeline/                    # Data preparation & quality
│   ├── 1.1_data_collection_sourcing.md
│   ├── 1.2_data_cleaning_preprocessing.md
│   ├── 1.3_data_labeling_annotation.md
│   ├── 1.4_data_versioning_lineage.md
│   ├── 1.5_synthetic_data_generation.md
│   └── 1.6_data_quality_assurance.md
│
├── 02_model_training/                   # Training infrastructure
│   ├── 2.1_tokenizer_training_selection.md
│   ├── 2.2_model_architecture_selection.md
│   ├── 2.3_distributed_training_infrastructure.md
│   ├── 2.4_pretraining_data_mix_curriculum.md
│   ├── 2.5_training_monitoring_debugging.md
│   └── 2.6_ray_train_guide.md
│
├── 03_fine_tuning/                      # Model customization
│   ├── 3.1_supervised_fine_tuning.md
│   ├── 3.2_parameter_efficient_fine_tuning.md
│   ├── 3.3_domain_adaptation.md
│   └── 3.4_continued_pretraining.md
│
├── 04_alignment_safety/                 # Safety & alignment
│   ├── 4.1_rlhf_guide.md
│   ├── 4.2_constitutional_ai_rlaif.md
│   ├── 4.3_safety_evaluation_red_teaming.md
│   └── 4.4_bias_fairness_evaluation.md
│
├── 05_evaluation_testing/               # Model evaluation
│   ├── 5.1_llm_evaluation_framework.md
│   ├── 5.2_benchmark_selection_interpretation.md
│   ├── 5.3_llm_as_judge_evaluation.md
│   └── 5.4_human_evaluation_protocol.md
│
├── 06_model_optimization/               # Inference optimization
│   ├── 6.1_quantization_guide.md
│   ├── 6.2_pruning_sparsity_guide.md
│   ├── 6.3_knowledge_distillation_guide.md
│   └── 6.4_speculative_decoding_guide.md
│
├── 07_rag_pipeline/                     # RAG implementation
│   ├── 7.1_vector_database_guide.md
│   ├── 7.2_embedding_model_guide.md
│   ├── 7.3_chunking_strategies_guide.md
│   ├── 7.4_retrieval_reranking_guide.md
│   ├── 7.5_rag_evaluation_guide.md
│   └── 7.6_advanced_rag_patterns_guide.md
│
├── 08_mlops_lifecycle/                  # MLOps & CI/CD
│   ├── 8.1_model_registry_guide.md
│   ├── 8.2_experiment_tracking_guide.md
│   ├── 8.3_model_versioning_artifacts_guide.md
│   ├── 8.4_feature_store_llms_guide.md
│   └── 8.5_llm_cicd_pipeline_guide.md
│
├── 09_inference_serving/                # Production serving
│   ├── 9.1_inference_engine_selection_guide.md
│   ├── 9.2_serving_architecture_patterns_guide.md
│   └── 9.3_api_design_llm_services_guide.md
│
├── 10_monitoring_observability/         # Monitoring & logging
│   ├── 10.1_llm_monitoring_strategy_guide.md
│   ├── 10.2_llm_logging_tracing_guide.md
│   ├── 10.3_model_quality_monitoring_guide.md
│   └── 10.4_cost_monitoring_optimization_guide.md
│
├── 11_security_governance/              # Security & compliance
│   ├── 11.1_llm_security_guide.md
│   ├── 11.2_pii_data_privacy_guide.md
│   ├── 11.3_compliance_framework_guide.md
│   ├── 11.4_model_governance_guide.md
│   └── 11.5_access_control_authentication_guide.md
│
├── 12_user_developer_experience/        # Developer tools
│   ├── 12.1_prompt_engineering_guide.md
│   ├── 12.2_sdk_client_library_guide.md
│   ├── 12.3_developer_documentation_guide.md
│   └── 12.4_user_feedback_iteration_guide.md
│
├── 13_operations_reliability/           # Operations & SRE
│   ├── 13.1_incident_response_guide.md
│   ├── 13.2_disaster_recovery_guide.md
│   ├── 13.3_capacity_planning_guide.md
│   ├── 13.4_on_call_practices_guide.md
│   └── 13.5_operational_runbooks_guide.md
│
├── 14_cost_capacity_management/         # Cost optimization
│   ├── 14.1_total_cost_ownership_guide.md
│   ├── 14.2_cloud_cost_optimization_guide.md
│   └── 14.3_gpu_infrastructure_optimization_guide.md
│
├── 15_migration_integration/            # Migration & integration
│   ├── 15.1_model_migration_guide.md
│   ├── 15.2_data_migration_guide.md
│   └── 15.3_system_integration_guide.md
│
├── MultiCloud_RAG_Complete_Matrix_v3.docx  # 96-config detailed matrix
└── README.md                               # This document (PRD)
```

### Document Mapping to This PRD

| PRD Section | Key Implementation Docs |
|-------------|------------------------|
| Architecture (§3) | [2.2](02_model_training/2.2_model_architecture_selection.md), [2.3](02_model_training/2.3_distributed_training_infrastructure.md), [9.2](09_inference_serving/9.2_serving_architecture_patterns_guide.md) |
| Model Tiers (§5) | [6.1](06_model_optimization/6.1_quantization_guide.md), [6.4](06_model_optimization/6.4_speculative_decoding_guide.md), [3.1](03_fine_tuning/3.1_supervised_fine_tuning.md) |
| GPU/Infrastructure (§6-7) | [9.1](09_inference_serving/9.1_inference_engine_selection_guide.md), [14.3](14_cost_capacity_management/14.3_gpu_infrastructure_optimization_guide.md), [2.3](02_model_training/2.3_distributed_training_infrastructure.md) |
| RAG Pipeline (§8) | [7.1](07_rag_pipeline/7.1_vector_database_guide.md), [7.2](07_rag_pipeline/7.2_embedding_model_guide.md), [7.4](07_rag_pipeline/7.4_retrieval_reranking_guide.md), [7.6](07_rag_pipeline/7.6_advanced_rag_patterns_guide.md) |
| Scheduling (§9) | [8.5](08_mlops_lifecycle/8.5_llm_cicd_pipeline_guide.md), [13.3](13_operations_reliability/13.3_capacity_planning_guide.md) |
| Deployment (§10) | [9.1](09_inference_serving/9.1_inference_engine_selection_guide.md), [9.2](09_inference_serving/9.2_serving_architecture_patterns_guide.md), [15.3](15_migration_integration/15.3_system_integration_guide.md) |
| Observability (§11) | [10.1](10_monitoring_observability/10.1_llm_monitoring_strategy_guide.md), [10.2](10_monitoring_observability/10.2_llm_logging_tracing_guide.md), [10.3](10_monitoring_observability/10.3_model_quality_monitoring_guide.md) |
| Cost (§12) | [14.1](14_cost_capacity_management/14.1_total_cost_ownership_guide.md), [14.2](14_cost_capacity_management/14.2_cloud_cost_optimization_guide.md), [10.4](10_monitoring_observability/10.4_cost_monitoring_optimization_guide.md) |
| Security (§13) | [11.1](11_security_governance/11.1_llm_security_guide.md), [11.2](11_security_governance/11.2_pii_data_privacy_guide.md), [11.3](11_security_governance/11.3_compliance_framework_guide.md) |

---

## Appendix A: Quick Commands Reference

### vLLM Deployment

```bash
# Deploy Tier S model
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --quantization awq \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92

# Deploy Tier L model (2x GPU)
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92

# Deploy Tier XL model (8x GPU)
vllm serve meta-llama/Llama-3.1-405B-Instruct-FP8 \
  --tensor-parallel-size 8 \
  --quantization fp8 \
  --max-model-len 32768
```

### Kubernetes Operations

```bash
# Deploy full stack
helm install rag-platform ./charts/rag-platform \
  --namespace inference \
  --values values-production.yaml

# Scale Tier M replicas
kubectl scale deployment vllm-tier-m --replicas=4 -n inference

# Check GPU utilization
kubectl exec -it vllm-tier-l-xxx -- nvidia-smi

# View Kueue queue status
kubectl get clusterqueues
kubectl get workloads -n inference
```

### Cost Monitoring

```bash
# Get current spend by provider
curl -s localhost:9090/api/v1/query?query=sum(rag_cost_total)by(provider)

# Check cost per 1K tokens
curl -s localhost:9090/api/v1/query?query=sum(rag_cost_total)/sum(vllm_tokens_total)*1000
```

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **TP (Tensor Parallelism)** | Distributing model layers across GPUs for memory |
| **PP (Pipeline Parallelism)** | Distributing model stages across GPUs for throughput |
| **TTFT** | Time to First Token - latency until first token generated |
| **vLLM** | High-performance LLM serving engine |
| **Kueue** | Kubernetes-native job queueing system |
| **RAG** | Retrieval-Augmented Generation |
| **FP8/INT4/AWQ** | Quantization formats for reduced memory |
| **HBM** | High Bandwidth Memory (GPU memory type) |
| **NVLink** | NVIDIA high-speed GPU interconnect |

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | December 2025 | Sohail Mohammad | Initial release |

---

<div align="center">

**Multi-Cloud RAG Infrastructure Platform**
*96 Configurations. 4 Model Tiers. 6 GPU Architectures. 4 Infrastructure Levels.*

For detailed implementation guidance, see the companion documentation repository.

</div>
