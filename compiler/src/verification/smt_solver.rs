// SMT Solver Integration
// Interfaces with Z3 or similar SMT solvers for automated theorem proving

use crate::parser::ast::*;
use crate::error::{Error, Result};
use std::process::{Command, Stdio};
use std::io::Write;

// SMT solver interface
pub struct SmtSolver {
    solver_type: SolverType,
    timeout: u32,
}

#[derive(Debug, Clone, Copy)]
pub enum SolverType {
    Z3,
    CVC5,
    Yices,
}

impl SmtSolver {
    pub fn new() -> Self {
        SmtSolver {
            solver_type: SolverType::Z3,
            timeout: 5000, // 5 seconds
        }
    }
    
    pub fn with_solver(solver_type: SolverType) -> Self {
        SmtSolver {
            solver_type,
            timeout: 5000,
        }
    }
    
    pub fn set_timeout(&mut self, timeout_ms: u32) {
        self.timeout = timeout_ms;
    }
    
    // Prove an expression using SMT solver
    pub fn prove(&self, expr: &Expr) -> Result<bool> {
        // Convert expression to SMT-LIB format
        let smt_lib = self.to_smt_lib(expr)?;
        
        // Call SMT solver
        let result = self.call_solver(&smt_lib)?;
        
        Ok(result)
    }
    
    // Check satisfiability
    pub fn check_sat(&self, expr: &Expr) -> Result<SatResult> {
        let smt_lib = self.to_smt_lib(expr)?;
        let output = self.call_solver_raw(&smt_lib)?;
        
        if output.contains("unsat") {
            Ok(SatResult::Unsat)
        } else if output.contains("sat") {
            Ok(SatResult::Sat)
        } else {
            Ok(SatResult::Unknown)
        }
    }
    
    // Get model for satisfiable formula
    pub fn get_model(&self, expr: &Expr) -> Result<Option<Model>> {
        let smt_lib = format!("{}\n(get-model)", self.to_smt_lib(expr)?);
        let output = self.call_solver_raw(&smt_lib)?;
        
        if output.contains("sat") {
            // Parse model from output
            Ok(Some(Model::new()))
        } else {
            Ok(None)
        }
    }
    
    // Convert VeZ expression to SMT-LIB format
    fn to_smt_lib(&self, expr: &Expr) -> Result<String> {
        let mut smt = String::new();
        
        // SMT-LIB header
        smt.push_str("(set-logic ALL)\n");
        smt.push_str("(set-option :timeout ");
        smt.push_str(&self.timeout.to_string());
        smt.push_str(")\n\n");
        
        // Declare variables
        let vars = self.collect_variables(expr);
        for var in vars {
            smt.push_str(&format!("(declare-const {} Int)\n", var));
        }
        smt.push_str("\n");
        
        // Assert the formula
        smt.push_str("(assert ");
        smt.push_str(&self.expr_to_smt(expr)?);
        smt.push_str(")\n\n");
        
        // Check satisfiability
        smt.push_str("(check-sat)\n");
        
        Ok(smt)
    }
    
    fn expr_to_smt(&self, expr: &Expr) -> Result<String> {
        match expr {
            Expr::Literal { value, .. } => {
                match value {
                    Literal::Int(n) => Ok(n.to_string()),
                    Literal::Bool(b) => Ok(b.to_string()),
                    _ => Err(Error::new("Unsupported literal type for SMT", Span::dummy())),
                }
            }
            Expr::Variable { name, .. } => Ok(name.clone()),
            Expr::Binary { op, left, right, .. } => {
                let left_smt = self.expr_to_smt(left)?;
                let right_smt = self.expr_to_smt(right)?;
                
                let op_smt = match op {
                    BinaryOp::Add => "+",
                    BinaryOp::Sub => "-",
                    BinaryOp::Mul => "*",
                    BinaryOp::Div => "div",
                    BinaryOp::Eq => "=",
                    BinaryOp::Ne => "distinct",
                    BinaryOp::Lt => "<",
                    BinaryOp::Le => "<=",
                    BinaryOp::Gt => ">",
                    BinaryOp::Ge => ">=",
                    BinaryOp::And => "and",
                    BinaryOp::Or => "or",
                    _ => return Err(Error::new("Unsupported binary operator for SMT", Span::dummy())),
                };
                
                Ok(format!("({} {} {})", op_smt, left_smt, right_smt))
            }
            Expr::Unary { op, operand, .. } => {
                let operand_smt = self.expr_to_smt(operand)?;
                
                let op_smt = match op {
                    UnaryOp::Not => "not",
                    UnaryOp::Neg => "-",
                    _ => return Err(Error::new("Unsupported unary operator for SMT", Span::dummy())),
                };
                
                Ok(format!("({} {})", op_smt, operand_smt))
            }
            _ => Err(Error::new("Unsupported expression type for SMT", Span::dummy())),
        }
    }
    
    fn collect_variables(&self, expr: &Expr) -> Vec<String> {
        let mut vars = Vec::new();
        self.collect_variables_rec(expr, &mut vars);
        vars.sort();
        vars.dedup();
        vars
    }
    
    fn collect_variables_rec(&self, expr: &Expr, vars: &mut Vec<String>) {
        match expr {
            Expr::Variable { name, .. } => {
                vars.push(name.clone());
            }
            Expr::Binary { left, right, .. } => {
                self.collect_variables_rec(left, vars);
                self.collect_variables_rec(right, vars);
            }
            Expr::Unary { operand, .. } => {
                self.collect_variables_rec(operand, vars);
            }
            _ => {}
        }
    }
    
    fn call_solver(&self, smt_lib: &str) -> Result<bool> {
        let output = self.call_solver_raw(smt_lib)?;
        Ok(output.contains("unsat"))
    }
    
    fn call_solver_raw(&self, smt_lib: &str) -> Result<String> {
        let solver_cmd = match self.solver_type {
            SolverType::Z3 => "z3",
            SolverType::CVC5 => "cvc5",
            SolverType::Yices => "yices-smt2",
        };
        
        let mut child = Command::new(solver_cmd)
            .arg("-in")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|e| Error::new(format!("Failed to spawn SMT solver: {}", e), Span::dummy()))?;
        
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(smt_lib.as_bytes())
                .map_err(|e| Error::new(format!("Failed to write to SMT solver: {}", e), Span::dummy()))?;
        }
        
        let output = child.wait_with_output()
            .map_err(|e| Error::new(format!("Failed to read SMT solver output: {}", e), Span::dummy()))?;
        
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SatResult {
    Sat,
    Unsat,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct Model {
    pub assignments: Vec<(String, i64)>,
}

impl Model {
    pub fn new() -> Self {
        Model {
            assignments: Vec::new(),
        }
    }
    
    pub fn add_assignment(&mut self, var: String, value: i64) {
        self.assignments.push((var, value));
    }
    
    pub fn get_value(&self, var: &str) -> Option<i64> {
        self.assignments.iter()
            .find(|(v, _)| v == var)
            .map(|(_, val)| *val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_smt_solver_creation() {
        let solver = SmtSolver::new();
        assert_eq!(solver.timeout, 5000);
    }
    
    #[test]
    fn test_model_creation() {
        let mut model = Model::new();
        model.add_assignment("x".to_string(), 42);
        
        assert_eq!(model.get_value("x"), Some(42));
        assert_eq!(model.get_value("y"), None);
    }
}
