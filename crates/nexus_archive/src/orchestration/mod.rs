//! Archive Orchestration Module
//! 
//! Manages lifecycle of cold storage data across DNA, holographic, and 5D optical mediums.
//! Implements regime-based archival policies and cold retrieval APIs.

pub mod eternal_archive_manager;
pub mod regime_archival_policy;
pub mod cold_retrieval_api;

pub use eternal_archive_manager::EternalArchiveManager;
pub use regime_archival_policy::RegimeArchivalPolicy;
pub use cold_retrieval_api::ColdRetrievalApi;
