//! Diagnostics module for enhanced error reporting

pub mod suggestions;

pub use suggestions::{
    Suggestion, SuggestionEngine, SuggestionKind,
    ErrorFormatter, FunctionSignature,
    levenshtein_distance, get_error_description, get_error_code_name,
    ERROR_CODES,
};
