//! Prompt templates for AI code generation

use std::collections::HashMap;

/// A prompt template for code generation
#[derive(Debug, Clone)]
pub struct PromptTemplate {
    pub id: String,
    pub description: String,
    pub template: String,
    pub variables: Vec<String>,
}

impl PromptTemplate {
    pub fn render(&self, vars: &HashMap<String, String>) -> String {
        let mut result = self.template.clone();
        for var in &self.variables {
            if let Some(value) = vars.get(var) {
                result = result.replace(&format!("{{{}}}", var), value);
            }
        }
        result
    }
}

/// Library of prompt templates
pub struct PromptLibrary {
    templates: HashMap<String, PromptTemplate>,
}

impl PromptLibrary {
    pub fn new() -> Self {
        let mut lib = PromptLibrary {
            templates: HashMap::new(),
        };
        lib.add_defaults();
        lib
    }

    pub fn get(&self, id: &str) -> Option<&PromptTemplate> {
        self.templates.get(id)
    }

    pub fn all(&self) -> impl Iterator<Item = &PromptTemplate> {
        self.templates.values()
    }

    pub fn add(&mut self, template: PromptTemplate) {
        self.templates.insert(template.id.clone(), template);
    }

    fn add_defaults(&mut self) {
        self.add(PromptTemplate {
            id: "function".to_string(),
            description: "Generate a function from description".to_string(),
            template: r#"Generate a VeZ function that {description}.

Context:
{context}

Requirements:
- Use VeZ syntax (Python-like with type annotations)
- Include type annotations for all parameters and return type
- Add @doc annotation with description
- Handle edge cases appropriately

Generate the function:"#.to_string(),
            variables: vec!["description".to_string(), "context".to_string()],
        });

        self.add(PromptTemplate {
            id: "struct".to_string(),
            description: "Generate a struct from description".to_string(),
            template: r#"Generate a VeZ struct for {description}.

Context:
{context}

Requirements:
- Use VeZ struct syntax
- Include appropriate fields with types
- Add visibility modifiers (pub where appropriate)
- Include @doc annotation

Generate the struct:"#.to_string(),
            variables: vec!["description".to_string(), "context".to_string()],
        });

        self.add(PromptTemplate {
            id: "impl".to_string(),
            description: "Generate an impl block for a type".to_string(),
            template: r#"Generate VeZ methods for the {type_name} type.

Context:
{context}

Existing type:
{type_def}

Methods to implement:
{methods}

Generate the impl block:"#.to_string(),
            variables: vec![
                "type_name".to_string(),
                "context".to_string(),
                "type_def".to_string(),
                "methods".to_string(),
            ],
        });

        self.add(PromptTemplate {
            id: "test".to_string(),
            description: "Generate tests for a function".to_string(),
            template: r#"Generate tests for this VeZ function:

```zari
{function}
```

Generate comprehensive tests covering:
- Happy path scenarios
- Edge cases
- Error conditions
- Boundary values

Use the @test annotation for test functions."#.to_string(),
            variables: vec!["function".to_string()],
        });

        self.add(PromptTemplate {
            id: "gpu_kernel".to_string(),
            description: "Generate a GPU kernel".to_string(),
            template: r#"Generate a VeZ GPU kernel that {description}.

Context:
{context}

Requirements:
- Use @gpu annotation with thread configuration
- Use VeZ GPU intrinsics (@threadIdx, @blockIdx, @blockDim)
- Optimize for parallel execution
- Handle boundary conditions

Generate the kernel:"#.to_string(),
            variables: vec!["description".to_string(), "context".to_string()],
        });

        self.add(PromptTemplate {
            id: "async_fn".to_string(),
            description: "Generate an async function".to_string(),
            template: r#"Generate an async VeZ function that {description}.

Context:
{context}

Requirements:
- Use async/await syntax
- Handle errors with Result type
- Include proper error types
- Consider cancellation safety

Generate the function:"#.to_string(),
            variables: vec!["description".to_string(), "context".to_string()],
        });

        self.add(PromptTemplate {
            id: "refactor".to_string(),
            description: "Refactor code to improve quality".to_string(),
            template: r#"Refactor this VeZ code to {goal}:

```zari
{code}
```

Requirements:
- Maintain exact same behavior
- Improve {aspect}
- Follow VeZ best practices

Refactored code:"#.to_string(),
            variables: vec!["goal".to_string(), "code".to_string(), "aspect".to_string()],
        });

        self.add(PromptTemplate {
            id: "doc".to_string(),
            description: "Generate documentation".to_string(),
            template: r#"Generate documentation for this VeZ code:

```zari
{code}
```

Generate:
1. Module-level documentation
2. Function/struct documentation with examples
3. Parameter descriptions

Use @doc annotations."#.to_string(),
            variables: vec!["code".to_string()],
        });
    }

    pub fn search(&self, query: &str) -> Vec<&PromptTemplate> {
        let query = query.to_lowercase();
        self.templates
            .values()
            .filter(|t| {
                t.id.to_lowercase().contains(&query)
                    || t.description.to_lowercase().contains(&query)
            })
            .collect()
    }
}

impl Default for PromptLibrary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_render() {
        let tmpl = PromptTemplate {
            id: "test".to_string(),
            description: "Test template".to_string(),
            template: "Hello {name}!".to_string(),
            variables: vec!["name".to_string()],
        };
        
        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "World".to_string());
        
        assert_eq!(tmpl.render(&vars), "Hello World!");
    }

    #[test]
    fn test_library_defaults() {
        let lib = PromptLibrary::new();
        assert!(lib.get("function").is_some());
        assert!(lib.get("struct").is_some());
        assert!(lib.get("test").is_some());
    }
}
