//! Stage 9: High-Performance NLP, LLM Inference Bridge & Alternative Data Ingestion
//! 
//! This crate provides zero-copy text processing, LLM inference bridging,
//! and real-time knowledge graph traversal for alpha generation.

pub mod ingestion;
pub mod tokenization;
pub mod inference;
pub mod graph;
pub mod alpha;

pub use ingestion::async_stream_consumer::*;
pub use ingestion::lock_free_bloom_filter::*;
pub use tokenization::zero_copy_simd_tokenizer::*;
pub use inference::trt_llm_ffi_bridge::*;
pub use inference::continuous_batching_queue::*;
pub use inference::paged_attention_pool::*;
pub use graph::entity_ticker_mapper::*;
pub use graph::sharded_knowledge_graph::*;
pub use graph::event_propagation::*;
pub use alpha::hawkish_dovish_scorer::*;
pub use alpha::sentiment_decay::*;
pub use alpha::nlp_alpha_fusion::*;
