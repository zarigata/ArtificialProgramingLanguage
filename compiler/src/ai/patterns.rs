//! Code pattern matching for AI suggestions

use std::collections::HashMap;

/// A recognized code pattern
#[derive(Debug, Clone)]
pub struct CodePattern {
    pub id: String,
    pub name: String,
    pub description: String,
    pub template: String,
    pub category: PatternCategory,
    pub complexity: PatternComplexity,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PatternCategory {
    Algorithm,
    DataStructure,
    Concurrency,
    ErrorHandling,
    Resource,
    Optimization,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PatternComplexity {
    Basic,
    Intermediate,
    Advanced,
}

/// Pattern matcher for code analysis
pub struct PatternMatcher {
    patterns: HashMap<String, CodePattern>,
}

impl PatternMatcher {
    pub fn new() -> Self {
        let mut matcher = PatternMatcher {
            patterns: HashMap::new(),
        };
        matcher.add_defaults();
        matcher
    }

    pub fn match_code(&self, code: &str) -> Vec<PatternMatch> {
        let mut matches = Vec::new();
        
        for (id, pattern) in &self.patterns {
            if self.matches_pattern(code, pattern) {
                matches.push(PatternMatch {
                    pattern_id: id.clone(),
                    confidence: 0.8,
                    suggestions: self.generate_suggestions(pattern),
                });
            }
        }
        
        matches
    }

    fn matches_pattern(&self, code: &str, pattern: &CodePattern) -> bool {
        let code_lower = code.to_lowercase();
        let template_lower = pattern.template.to_lowercase();
        
        let keywords: Vec<&str> = template_lower.split_whitespace().collect();
        let mut matched = 0;
        
        for keyword in &keywords {
            if code_lower.contains(keyword) {
                matched += 1;
            }
        }
        
        matched as f32 / keywords.len() as f32 > 0.5
    }

    fn generate_suggestions(&self, pattern: &CodePattern) -> Vec<String> {
        match pattern.category {
            PatternCategory::Algorithm => vec![
                format!("Consider using {} pattern", pattern.name),
                "Add input validation".to_string(),
                "Consider edge cases".to_string(),
            ],
            PatternCategory::Concurrency => vec![
                "Ensure thread safety".to_string(),
                "Consider using async/await".to_string(),
                "Add proper synchronization".to_string(),
            ],
            PatternCategory::ErrorHandling => vec![
                "Handle all error cases".to_string(),
                "Provide meaningful error messages".to_string(),
                "Consider error propagation".to_string(),
            ],
            _ => vec!["Follow best practices".to_string()],
        }
    }

    fn add_defaults(&mut self) {
        self.add(CodePattern {
            id: "iteration".to_string(),
            name: "Iteration".to_string(),
            description: "Loop over collection".to_string(),
            template: "for in range len iter".to_string(),
            category: PatternCategory::Algorithm,
            complexity: PatternComplexity::Basic,
        });

        self.add(CodePattern {
            id: "map_transform".to_string(),
            name: "Map Transform".to_string(),
            description: "Transform collection elements".to_string(),
            template: "map transform apply each element".to_string(),
            category: PatternCategory::Algorithm,
            complexity: PatternComplexity::Basic,
        });

        self.add(CodePattern {
            id: "filter_select".to_string(),
            name: "Filter Selection".to_string(),
            description: "Filter collection by predicate".to_string(),
            template: "filter where if condition select".to_string(),
            category: PatternCategory::Algorithm,
            complexity: PatternComplexity::Basic,
        });

        self.add(CodePattern {
            id: "error_result".to_string(),
            name: "Error Handling".to_string(),
            description: "Handle errors with Result type".to_string(),
            template: "result ok err match handle error".to_string(),
            category: PatternCategory::ErrorHandling,
            complexity: PatternComplexity::Basic,
        });

        self.add(CodePattern {
            id: "async_await".to_string(),
            name: "Async Operation".to_string(),
            description: "Asynchronous operation pattern".to_string(),
            template: "async await future spawn task".to_string(),
            category: PatternCategory::Concurrency,
            complexity: PatternComplexity::Intermediate,
        });

        self.add(CodePattern {
            id: "mutex_lock".to_string(),
            name: "Mutex Lock".to_string(),
            description: "Thread-safe access with mutex".to_string(),
            template: "mutex lock unlock guard sync".to_string(),
            category: PatternCategory::Concurrency,
            complexity: PatternComplexity::Intermediate,
        });

        self.add(CodePattern {
            id: "resource_cleanup".to_string(),
            name: "Resource Cleanup".to_string(),
            description: "Ensure resource cleanup".to_string(),
            template: "with resource drop close cleanup defer".to_string(),
            category: PatternCategory::Resource,
            complexity: PatternComplexity::Intermediate,
        });

        self.add(CodePattern {
            id: "memoization".to_string(),
            name: "Memoization".to_string(),
            description: "Cache function results".to_string(),
            template: "cache memoize store result".to_string(),
            category: PatternCategory::Optimization,
            complexity: PatternComplexity::Advanced,
        });
    }

    pub fn add(&mut self, pattern: CodePattern) {
        self.patterns.insert(pattern.id.clone(), pattern);
    }

    pub fn get(&self, id: &str) -> Option<&CodePattern> {
        self.patterns.get(id)
    }

    pub fn by_category(&self, category: PatternCategory) -> Vec<&CodePattern> {
        self.patterns
            .values()
            .filter(|p| p.category == category)
            .collect()
    }
}

impl Default for PatternMatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of pattern matching
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub pattern_id: String,
    pub confidence: f32,
    pub suggestions: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_matcher() {
        let matcher = PatternMatcher::new();
        
        let matches = matcher.match_code("for item in items: process(item)");
        assert!(!matches.is_empty());
    }

    #[test]
    fn test_by_category() {
        let matcher = PatternMatcher::new();
        
        let concurrency = matcher.by_category(PatternCategory::Concurrency);
        assert!(!concurrency.is_empty());
    }
}
