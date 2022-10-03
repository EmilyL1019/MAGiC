//! A `Metric` allows for calculating distances between instances in a `Dataset`.

use std::collections::HashSet;
use std::convert::TryInto;
use std::sync::Arc;
use num_traits::NumCast;
use crate::algorithms_lib;
use crate::Number;

/// A `Metric` is a function that takes two instances (generic over a `Number` T)
/// from a `Dataset` and deterministically produces a non-negative `Number` U.
///
/// Optionally, a `Metric` also allows us to encode one instance in terms of another
/// and decode decode an instance from a reference and an encoding.
pub trait Metric<T, U>: Send + Sync {
    /// Returns the name of the `Metric` as a String.
    fn name(&self) -> String;

    /// Returns the distance between two instances.
    fn distance(&self, x: &[T], y: &[T]) -> U;

    /// Encodes the target instance in terms of the reference and produces a vec of bytes.
    ///
    /// This method is optional and so the default just returns an Err.
    #[allow(unused_variables)]
    fn encode(&self, reference: &[T], target: &[T]) -> Result<Vec<u8>, String> {
        Err(format!("encode is not implemented for {:?}", self.name()))
    }

    /// Decodes a target instances from a reference instance and a bytes encoding.
    ///
    /// This method is optional and so the default just returns an Err.
    #[allow(unused_variables)]
    fn decode(&self, reference: &[T], encoding: &[u8]) -> Result<Vec<T>, String> {
        Err(format!("decode is not implemented for {:?}", self.name()))
    }
}

/// Returns a `Metric` from a given name, or an Err if the name
/// is not found among the implemented `Metrics`.
///
/// # Arguments
///
/// * `metric`: A `&str` name of a distance function.
/// This can be one of:
///   - "euclidean": L2-norm.
///   - "euclideansq": Squared L2-norm.
///   - "manhattan": L1-norm.
///   - "cosine": Cosine distance.
///   - "hamming": Hamming distance.
///   - "jaccard": Jaccard distance.
///
/// We plan on adding the following:
///   - "levenshtein": Edit-distance among strings (e.g. genomic/amino-acid sequences).
///   - "wasserstein": Earth-Mover-Distance among high-dimensional probability distributions (will be usable with images)
///   - "tanamoto": Jaccard distance between the Maximal-Common-Subgraph of two molecular structures.
pub fn metric_from_name<T: Number, U: Number>(metric: &str) -> Result<Arc<dyn Metric<T, U>>, String> {
    match metric {
        "euclidean" => Ok(Arc::new(Euclidean)),
        "euclideansq" => Ok(Arc::new(EuclideanSq)),
        "manhattan" => Ok(Arc::new(Manhattan)),
        "cosine" => Ok(Arc::new(Cosine)),
        "hamming" => Ok(Arc::new(Hamming)),
        "jaccard" => Ok(Arc::new(Jaccard)),
        _ => Err(format!("{} is not defined as a metric.", metric)),
    }
}

/// Implements Euclidean distance, the L2-norm.
pub struct Euclidean;

impl<T: Number, U: Number> Metric<T, U> for Euclidean {
    fn name(&self) -> String {
        "euclidean".to_string()
    }

    fn distance(&self, x: &[T], y: &[T]) -> U {
        let d: T = x.iter().zip(y.iter()).map(|(&a, &b)| (a - b) * (a - b)).sum();
        let d: f64 = NumCast::from(d).unwrap();
        U::from(d.sqrt()).unwrap()
    }
}

/// Implements Squared-Euclidean distance, the squared L2-norm.
pub struct EuclideanSq;

impl<T: Number, U: Number> Metric<T, U> for EuclideanSq {
    fn name(&self) -> String {
        "euclideansq".to_string()
    }

    fn distance(&self, x: &[T], y: &[T]) -> U {
        let d: T = x.iter().zip(y.iter()).map(|(&a, &b)| (a - b) * (a - b)).sum();
        U::from(d).unwrap()
    }
}

/// Implements Manhattan/Cityblock distance, the L1-norm.
pub struct Manhattan;

impl<T: Number, U: Number> Metric<T, U> for Manhattan {
    fn name(&self) -> String {
        "manhattan".to_string()
    }

    fn distance(&self, x: &[T], y: &[T]) -> U {
        let d: T = x
            .iter()
            .zip(y.iter())
            .map(|(&a, &b)| if a > b { a - b } else { b - a })
            .sum();
        U::from(d).unwrap()
    }
}

/// Implements Cosine distance, 1 - cosine-similarity.
pub struct Cosine;

fn dot<T: Number>(x: &[T], y: &[T]) -> T {
    x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum()
}

impl<T: Number, U: Number> Metric<T, U> for Cosine {
    fn name(&self) -> String {
        "cosine".to_string()
    }

    #[allow(clippy::suspicious_operation_groupings)]
    fn distance(&self, x: &[T], y: &[T]) -> U {
        let xx = dot(x, x);
        if xx == T::zero() {
            return U::one();
        }

        let yy = dot(y, y);
        if yy == T::zero() {
            return U::one();
        }

        let xy = dot(x, y);
        if xy <= T::zero() {
            return U::one();
        }

        let similarity: f64 = NumCast::from(xy * xy / (xx * yy)).unwrap();
        U::one() - U::from(similarity.sqrt()).unwrap()
    }
}

/// Implements Hamming distance.
/// This is not normalized by the number of features.
pub struct Hamming;

impl<T: Number, U: Number> Metric<T, U> for Hamming {
    fn name(&self) -> String {
        "hamming".to_string()
    }

    fn distance(&self, x: &[T], y: &[T]) -> U {
        let d = x.iter().zip(y.iter()).filter(|(&a, &b)| a != b).count();
        U::from(d).unwrap()
    }

    fn encode(&self, x: &[T], y: &[T]) -> Result<Vec<u8>, String> {
        let encoding = x
            .iter()
            .zip(y.iter())
            .enumerate()
            .filter(|(_, (&l, &r))| l != r)
            .flat_map(|(i, (_, &r))| {
                let mut i = (i as u64).to_be_bytes().to_vec();
                i.append(&mut r.to_bytes());
                i
            })
            .collect();
        Ok(encoding)
    }

    fn decode(&self, x: &[T], y: &[u8]) -> Result<Vec<T>, String> {
        let mut x = x.to_owned();
        let step = (8 + T::num_bytes()) as usize;
        y.chunks(step).for_each(|chunk| {
            let (index, value) = chunk.split_at(std::mem::size_of::<u64>());
            let index = u64::from_be_bytes(index.try_into().unwrap()) as usize;
            x[index] = T::from_bytes(value);
        });
        Ok(x)
    }
}

/// Implements Cosine distance, 1 - jaccard-similarity.
///
/// Warning: DO NOT use this with floating-point numbers.
pub struct Jaccard;

impl<T: Number, U: Number> Metric<T, U> for Jaccard {
    fn name(&self) -> String {
        "jaccard".to_string()
    }

    fn distance(&self, x: &[T], y: &[T]) -> U {
        if x.is_empty() || y.is_empty() {
            return U::one();
        }

        let x: HashSet<u64> = x.iter().map(|&a| NumCast::from(a).unwrap()).collect();
        let intersect = y.iter().filter(|&&b| x.contains(&NumCast::from(b).unwrap())).count();

        if intersect == x.len() && intersect == y.len() {
            return U::zero();
        }

        U::one() - U::from(intersect).unwrap() / U::from(x.len() + y.len() - intersect).unwrap()
    }
}

// Implements Needleman-Waterman
pub struct Needleman_Wunsch;

impl<T: String, U: String> Metric<T, U> for Needleman_Wunsch{
    fn name(&self) -> String {
        "needleman-wunsch".to_string()
    }

    fn distance(&self, x: &[T], y: &[T]) -> U {
        if x.is_empty() || y.is_empty() {
            return U::one();
        }

        let mut xStr: String = flat_map(x.iter(), |c| c.chars());
        let mut yStr: String = flat_map(y.iter(), |c| c.chars());
        // Modify the original Needleman_Wunsch to have one function complete the entire process
        algorithms_lib::Needleman_Wunsch::align(xStr, yStr);
    }
}

// Implements Smith-Waterman
pub struct Smith_Waterman;

impl<T: String, U: String> Metric<T, U> for Smith_Waterman{
    fn name(&self) -> String {
        "smith-waterman".to_string()
    }

    fn distance(&self, x: &[T], y: &[T]) -> U {
        if x.is_empty() || y.is_empty() {
            return U::one();
        }

        let mut xStr: String = flat_map(x.iter(), |c| c.chars());
        let mut yStr: String = flat_map(y.iter(), |c| c.chars());
        // Modify the original Needleman_Wunsch to have one function complete the entire process
        algorithms_lib::Smith_Waterman::align(xStr, yStr);
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::approx_eq;
    use ndarray::{arr2, Array2};

    use crate::metric::metric_from_name;

    #[test]
    fn test_on_real() {
        let data: Array2<f64> = arr2(&[[1., 2., 3.], [3., 3., 1.]]);
        let row0 = data.row(0).to_vec();
        let row1 = data.row(1).to_vec();

        let metric = metric_from_name("euclideansq").unwrap();
        approx_eq!(f64, metric.distance(&row0, &row0), 0.);
        approx_eq!(f64, metric.distance(&row0, &row1), 9.);

        let metric = metric_from_name("euclidean").unwrap();
        approx_eq!(f64, metric.distance(&row0, &row0), 0.);
        approx_eq!(f64, metric.distance(&row0, &row1), 3.);

        let metric = metric_from_name("manhattan").unwrap();
        approx_eq!(f64, metric.distance(&row0, &row0), 0.);
        approx_eq!(f64, metric.distance(&row0, &row1), 5.);
    }

    #[test]
    #[should_panic]
    fn test_panic() {
        let _ = metric_from_name::<f32, f32>("aloha").unwrap();
    }

    #[test]
    fn Needleman_Wunsch_Test1(){
        let data: Vec<String> = vec!["GTCAGGATCT".to_string(), "ATCAAGGCCA".to_string()];
        let seq1 = data[0];
        let seq2 = data[1];

        let metric =  metric_from_name("needleman-wunsch").unwrap();
        let (aligned_seq1, aligned_seq2, score) = metric.distance(x, y);

        assert_eq!(aligned_seq1, vec!["GTC-AGGATCT", "GTCA-GGATCT", "GTC-AGGATCT", "GTCA-GGATCT"]);
        assert_eq!(aligned_seq2, vec!["ATCAAGG-CCA", "ATCAAGG-CCA", "ATCAAGGC-CA", "ATCAAGGC-CA"]);
        assert_eq!(score, 1);
    }

    #[test]
    fn Needleman_Wunsch_Test2(){
        let data: Vec<String> = vec!["ATGCAGGA".to_string(), "CTGAA".to_string()];
        let seq1 = data[0];
        let seq2 = data[1];

        let metric =  metric_from_name("needleman-wunsch").unwrap();
        let (aligned_seq1, aligned_seq2, score) = metric.distance(x, y);

        assert_eq!(aligned_seq1, vec!["ATGCAGGA"]);
        assert_eq!(aligned_seq2, vec!["CTG-A--A"]);
        assert_eq!(score, 0);
    }

    #[test]
    fn Needleman_Wunsch_Test3(){
        let data: Vec<String> = vec!["AAGTAAGGTGCAGAATGAAA".to_string(), "CATTCAGGAAGCTGT".to_string()];
        let seq1 = data[0];
        let seq2 = data[1];

        let metric =  metric_from_name("needleman-wunsch").unwrap();
        let (aligned_seq1, aligned_seq2, score) = metric.distance(x, y);

        assert_eq!(aligned_seq1, vec!["AAGTAAGGTGCAGAATGAAA", "AAGTAAGGTGCAGAATGAAA", "AAGTAAGGTGCAGAATGAAA", "AAGTAAGGTGCAGAATGAAA", "AAGTAAGGTGCAGAATGAAA", "AAGTAAGGTGCAGAATGAAA",
        "AAGTAAGGTGCAGAATGAAA", "AAGTAAGGTGCAGAATGAAA", "AAGTAAGGTGCAGAATGAAA", "AAGTAAGGTGCAGAATGAAA", "AAGTAAGGTGCAGAATGAAA", "AAGTAAGGTGCAGAATGAAA",
        "AAGTAAGGTGCAGAATGAAA", "AAGTAAGGTGCAGAATGAAA", "AAGTAAGGTGCAGAATGAAA", "AAGTAAGGTGCAGAATGAAA", "AAGTAAGGTGCAGAATGAAA", "AAGTAAGGTGCAGAATGAAA",
        "AAGTAAGGTGCAGAATGAAA", "AAGTAAGGTGCAGAATGAAA", "AAGTAAGGTGCAGAATGAAA", "AAGTAAGGTGCAGAATGAAA", "AAGTAAGGTGCAGAATGAAA", "AAGTAAGGTGCAGAATGAAA",
        "AAGTAAGGTGCAGAATGAAA", "AAGTAAGGTGCAGAATGAAA", "AAGTAAGGTGCAGAATGAAA", "AAGTAAGGTGCAGAATGAAA", "AAGTAAGGTGCAGAATGAAA", "AAGTAAGGTGCAGAATGAAA"]);
        assert_eq!(aligned_seq2, vec!["CATTCA-G-GAAG-CTG--T", "CATTCAG--GAAG-CTG--T", "CATTCAGG--AAG-CTG--T", "CATTCAGG-A-AG-CTG--T", "CATTCAGGA--AG-CTG--T", 
        "CATTCA-G-GAAGC-TG--T", "CATTCAG--GAAGC-TG--T", "CATTCAGG--AAGC-TG--T", "CATTCAGG-A-AGC-TG--T", "CATTCAGGA--AGC-TG--T", 
        "CATTCA-G-GAAG-CTG-T-", "CATTCAG--GAAG-CTG-T-", "CATTCAGG--AAG-CTG-T-", "CATTCAGG-A-AG-CTG-T-", "CATTCAGGA--AG-CTG-T-", 
        "CATTCA-G-GAAGC-TG-T-", "CATTCAG--GAAGC-TG-T-", "CATTCAGG--AAGC-TG-T-", "CATTCAGG-A-AGC-TG-T-", "CATTCAGGA--AGC-TG-T-", 
        "CATTCA-G-GAAG-CTGT--", "CATTCAG--GAAG-CTGT--", "CATTCAGG--AAG-CTGT--", "CATTCAGG-A-AG-CTGT--", "CATTCAGGA--AG-CTGT--",
        "CATTCA-G-GAAGC-TGT--", "CATTCAG--GAAGC-TGT--", "CATTCAGG--AAGC-TGT--", "CATTCAGG-A-AGC-TGT--", "CATTCAGGA--AGC-TGT--"]);
        assert_eq!(score, -2);
    }

    #[test]
    fn Needleman_Wunsch_Test4(){
        let data: Vec<String> = vec!["TGACTG".to_string(), "AAGGTACAA".to_string()];
        let seq1 = data[0];
        let seq2 = data[1];

        let metric =  metric_from_name("needleman-wunsch").unwrap();
        let (aligned_seq1, aligned_seq2, score) = metric.distance(x, y);
        
        assert_eq!(aligned_seq1, vec!["--TG-ACTG", "-T-G-ACTG", "T--G-ACTG", "-TG--ACTG", "T-G--ACTG"]);
        assert_eq!(aligned_seq2, vec!["AAGGTACAA", "AAGGTACAA", "AAGGTACAA", "AAGGTACAA", "AAGGTACAA"]);
        assert_eq!(score, -3);
    }   
    
    #[test]
    fn Needleman_Wunsch_Test5(){
        let data: Vec<String> = vec!["CTAGATGAG".to_string(), "TTCAGT".to_string()];
        let seq1 = data[0];
        let seq2 = data[1];

        let metric =  metric_from_name("needleman-wunsch").unwrap();
        let (aligned_seq1, aligned_seq2, score) = metric.distance(x, y);
        
        assert_eq!(aligned_seq1, vec!["CTAGATGAG-", "CT-AGATGAG"]);
        assert_eq!(aligned_seq2, vec!["-T---TCAGT", "TTCAG-T---"]);
        assert_eq!(score, -2);
    }

    #[test]
    fn Needleman_Wunsch_Test6(){
        let data: Vec<String> = vec!["TTTGATGT".to_string(), "AAACTACA".to_string()];
        let seq1 = data[0];
        let seq2 = data[1];

        let metric =  metric_from_name("needleman-wunsch").unwrap();
        let (aligned_seq1, aligned_seq2, score) = metric.distance(x, y);

        assert_eq!(aligned_seq1, vec!["TTTGA-T-GT", "TTTGA-T-GT", "TTTGA-T-GT", "TTTGA-T-GT", "TTTGA-T-GT", 
        "TTTGA-T-GT", "--TTTGATGT", "-T-TTGATGT", "T--TTGATGT", "-TT-TGATGT", 
        "T-T-TGATGT", "TT--TGATGT", "TTTGA-TG-T", "TTTGA-TG-T", "TTTGA-TG-T", 
        "TTTGA-TG-T", "TTTGA-TG-T", "TTTGA-TG-T", "--TTTGATGT", "-T-TTGATGT", 
        "T--TTGATGT", "-TT-TGATGT", "T-T-TGATGT", "TT--TGATGT", "TTTGA-TGT-", 
        "TTTGA-TGT-", "TTTGA-TGT-", "TTTGA-TGT-", "TTTGA-TGT-", "TTTGA-TGT-", 
        "--TTTGATGT", "-T-TTGATGT", "T--TTGATGT", "-TT-TGATGT", "T-T-TGATGT",
        "TT--TGATGT"]);
        assert_eq!(aligned_seq2, vec!["--AAACTACA", "-A-AACTACA", "A--AACTACA", "-AA-ACTACA", "A-A-ACTACA", 
        "AA--ACTACA", "AAACT-A-CA", "AAACT-A-CA", "AAACT-A-CA", "AAACT-A-CA", 
        "AAACT-A-CA", "AAACT-A-CA", "--AAACTACA", "-A-AACTACA", "A--AACTACA", 
        "-AA-ACTACA", "A-A-ACTACA", "AA--ACTACA", "AAACT-AC-A", "AAACT-AC-A",
        "AAACT-AC-A", "AAACT-AC-A", "AAACT-AC-A", "AAACT-AC-A", "--AAACTACA",
        "-A-AACTACA", "A--AACTACA", "-AA-ACTACA", "A-A-ACTACA", "AA--ACTACA", 
        "AAACT-ACA-", "AAACT-ACA-", "AAACT-ACA-", "AAACT-ACA-", "AAACT-ACA-",
        "AAACT-ACA-"]);
        assert_eq!(score, -6);
    }
    
    #[test]
    fn Needleman_Wunsch_Test7(){
        let data: Vec<String> = vec!["AAAAA".to_string(), "TTTTT".to_string()];
        let seq1 = data[0];
        let seq2 = data[1];

        let metric =  metric_from_name("needleman-wunsch").unwrap();
        let (aligned_seq1, aligned_seq2, score) = metric.distance(x, y);

        assert_eq!(aligned_seq1, vec!["AAAAA"]);
        assert_eq!(aligned_seq2, vec!["TTTTT"]);
        assert_eq!(score, -5);
    }      
}
  
