use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymError {
    Shape(String),
    InvalidArgument(String),
}

impl Display for SymError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            SymError::Shape(msg) => write!(f, "shape error: {msg}"),
            SymError::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
        }
    }
}

impl Error for SymError {}

pub type Result<T> = std::result::Result<T, SymError>;
