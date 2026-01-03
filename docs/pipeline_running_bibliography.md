# Pipeline Running Bibliography
**Last Updated:** December 31, 2025
**Project:** Nuclear Envelope Detection and Spline Refinement Pipeline

---

## ORGANIZATION SYSTEM

**Tags:**
- `[STAT]` - Statistical methods
- `[OPT]` - Optimization algorithms  
- `[IMG]` - Image processing
- `[MICR]` - Microscopy theory
- `[SPL]` - Spline theory
- `[GEOM]` - Computational geometry
- `[ML]` - Machine learning
- `[BIO]` - Cell biology
- `[CAL]` - Camera calibration

---

## CAMERA CALIBRATION & DETECTOR RESPONSIVITY

### Photon Transfer Curve Method

**Janesick, J.R. (2001)**. *Scientific Charge-Coupled Devices*. SPIE Press, Bellingham, WA.
- **Tags:** `[CAL]` `[MICR]`
- **Context:** Camera calibration via variance-mean relationship
- **Key Content:** Chapter 3: "Photon Transfer Curve"
  - Variance-mean analysis for gain/offset determination
  - Dark noise characterization
- **Implementation:** `responsivity.py` - `detector_responsivity_determ()`
- **Parameters Derived:**
  - Camera gain (e-/ADU)
  - Dark offset (ADU)
  - Read noise (photons)
- **Cross-Reference:** See Gonzalez & Woods (2008) for noise modeling

**Gonzalez, R.C. & Woods, R.E. (2008)**. *Digital Image Processing*, 3rd edition. Prentice Hall.
- **Tags:** `[IMG]` `[CAL]`
- **Context:** Digital image fundamentals
- **Key Content:** Chapter 5: "Image Restoration and Reconstruction"
  - Section 5.2: Noise models (Gaussian, Poisson, impulse)
  - Noise parameter estimation
- **Relevance:** Foundation for understanding detector noise characteristics
- **Cross-Reference:** See Janesick (2001) for CCD-specific methods

---

## STATISTICAL METHODS & HYPOTHESIS TESTING

### Likelihood Ratio Test & Model Selection

**Ober, R.J., Ram, S. & Ward, E.S. (2004)**. "Localization accuracy in single-molecule microscopy." *Biophysical Journal*, 86(2), 1185-1200.
- **Tags:** `[STAT]` `[MICR]` `[OPT]`
- **Context:** Statistical foundations for single-molecule localization
- **Key Content:**
  - Cramér-Rao Lower Bound (CRLB) for localization precision
  - Maximum likelihood estimation framework
- **Implementation:** Theoretical foundation for precision estimates
- **Cross-Reference:** See Smith et al. (2010) for likelihood ratio testing

**Smith, C.S., Joseph, N., Rieger, B. & Lidke, K.A. (2010)**. "Fast, single-molecule localization that achieves theoretically minimum uncertainty." *Nature Methods*, 7(5), 373-375.
- **Tags:** `[STAT]` `[MICR]` `[OPT]`
- **Context:** Likelihood ratio test for quality control
- **Key Content:**
  - GLRT framework for detection and fitting
  - Statistical hypothesis testing for complex models
  - Quality metrics for fitted parameters
- **Implementation:** `npc_spline_refinement.py` - LRT for outlier detection
- **Replaces:** Hard-coded 5% Poisson threshold (statistically unjustified)
- **Cross-Reference:** See Ober et al. (2004) for theoretical foundations

### Akaike Information Criterion

**Akaike, H. (1974)**. "A new look at the statistical model identification." *IEEE Transactions on Automatic Control*, 19(6), 716-723.
- **Tags:** `[STAT]`
- **Context:** Model selection and comparison
- **Key Content:**
  - AIC = 2k - 2ln(L), where k = # parameters, L = likelihood
  - Penalizes model complexity
  - Lower AIC indicates better model fit
- **Implementation:** Used in LRT framework for nested model comparison
- **Decision Rule:** If ΔAIC > 2, reject complex model (exclude point)
- **Cross-Reference:** See Burnham & Anderson (2002) for practical applications

**Burnham, K.P. & Anderson, D.R. (2002)**. *Model Selection and Multimodel Inference: A Practical Information-Theoretic Approach*, 2nd edition. Springer.
- **Tags:** `[STAT]`
- **Context:** Practical AIC application
- **Key Content:**
  - Chapter 2: "Information and Likelihood Theory"
  - AIC weights and model averaging
- **Relevance:** Guidelines for interpreting AIC differences
- **Cross-Reference:** See Akaike (1974) for original theory

### Robust Statistics

**Huber, P.J. & Ronchetti, E.M. (2009)**. *Robust Statistics*, 2nd edition. Wiley.
- **Tags:** `[STAT]`
- **Context:** Comprehensive robust statistics theory
- **Key Content:**
  - M-estimators and their properties
  - Influence functions and breakdown points
- **Relevance:** Theoretical foundation for outlier-resistant methods
- **Cross-Reference:** See Rousseeuw & Croux (1993) for specific MAD applications

**Rousseeuw, P.J. & Croux, C. (1993)**. "Alternatives to the median absolute deviation." *Journal of the American Statistical Association*, 88(424), 1273-1283.
- **Tags:** `[STAT]`
- **Context:** Robust estimation of scale
- **Key Content:**
  - MAD (Median Absolute Deviation) as robust alternative to standard deviation
  - Sn and Qn estimators with higher efficiency
- **Implementation:** Registration stability analysis uses MAD-based filtering
- **Advantage:** Resistant to outliers (50% breakdown point)
- **Cross-Reference:** See Huber & Ronchetti (2009) for general robust methods

### False Discovery Rate Control

**Benjamini, Y. & Yekutieli, D. (2001)**. "The control of the false discovery rate in multiple testing under dependency." *Annals of Statistics*, 29(4), 1165-1188.
- **Tags:** `[STAT]`
- **Context:** Multiple testing correction with dependency
- **Key Content:**
  - BY procedure for FDR control under arbitrary dependency
  - More conservative than Benjamini-Hochberg
- **Implementation:** Potential use in registration stability filtering
- **Advantage:** Valid under positive or negative correlation between tests
- **Cross-Reference:** See Benjamini & Hochberg (1995) for independent case

---

## OPTIMIZATION ALGORITHMS

### Levenberg-Marquardt Algorithm

**Marquardt, D.W. (1963)**. "An algorithm for least-squares estimation of nonlinear parameters." *Journal of the Society for Industrial and Applied Mathematics*, 11(2), 431-441.
- **Tags:** `[OPT]` `[STAT]`
- **Context:** Nonlinear least squares optimization
- **Key Content:**
  - Damping parameter λ interpolates between gradient descent and Gauss-Newton
  - λ adjustment based on iteration success
  - Convergence theory for well-behaved problems
- **Implementation:** Core algorithm in `npc_spline_refinement.py` via Theseus
- **Adaptive Enhancement:** Custom step size adjustment prevents NaN failures
- **Cross-Reference:** See Dennis & Schnabel (1996) for convergence criteria

**Dennis, J.E. & Schnabel, R.B. (1996)**. *Numerical Methods for Unconstrained Optimization and Nonlinear Equations*. SIAM.
- **Tags:** `[OPT]` `[STAT]`
- **Context:** Comprehensive optimization theory
- **Key Content:**
  - Chapter 8: "Stopping Criteria" - Convergence tests
  - Chapter 10: "Nonlinear Least Squares" - LM algorithm analysis
- **Relevance:** Theoretical justification for parameter-specific tolerances
- **Cross-Reference:** See Marquardt (1963) for original LM method

**Gill, P.E., Murray, W. & Wright, M.H. (1981)**. *Practical Optimization*. Academic Press.
- **Tags:** `[OPT]`
- **Context:** Practical optimization methods
- **Key Content:**
  - Section 4.1: "Convergence Tests" - Multi-scale parameters
  - Line search and trust region methods
- **Implementation Notes:** Suggests position params need tighter tolerance than shape params
- **Cross-Reference:** See Dennis & Schnabel (1996) for detailed theory

### Weighted Least Squares

**Bevington, P.R. & Robinson, D.K. (2003)**. *Data Reduction and Error Analysis for the Physical Sciences*, 3rd edition. McGraw-Hill.
- **Tags:** `[STAT]` `[OPT]`
- **Context:** Experimental data analysis
- **Key Content:**
  - Chapter 6: "Least-Squares Fit to a Polynomial"
  - Chapter 8: "Least-Squares Fit to an Arbitrary Function"
  - Inverse-variance weighting for Poisson-distributed data
- **Implementation:** Poisson weighting in profile fitting (weight ∝ 1/I)
- **Justification:** Optimal weights for heteroscedastic data
- **Cross-Reference:** See Lupton (1993) for astronomical applications

**Lupton, R. (1993)**. *Statistics in Theory and Practice*. Princeton University Press.
- **Tags:** `[STAT]`
- **Context:** Statistical methods in astronomy/physics
- **Key Content:**
  - Chapter 7: "Fitting Data" - Weighted regression
  - Error propagation in nonlinear fits
- **Relevance:** Poisson noise handling in photon-counting detectors
- **Cross-Reference:** See Bevington & Robinson (2003) for practical examples

---

## MACHINE LEARNING & DEEP LEARNING

### U-Net Architecture

**Ronneberger, O., Fischer, P. & Brox, T. (2015)**. "U-Net: Convolutional networks for biomedical image segmentation." *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, Springer LNCS, 9351, 234-241.
- **Tags:** `[ML]` `[IMG]`
- **Context:** Original U-Net architecture for biomedical segmentation
- **Key Content:**
  - Encoder-decoder structure with skip connections
  - Data augmentation for limited training data
- **Relevance:** Foundation for U-Net++ used in initial NE detection
- **Cross-Reference:** See Zhou et al. (2018) for U-Net++ improvements

### U-Net++ Architecture

**Zhou, Z., Siddiquee, M.M.R., Tajbakhsh, N. & Liang, J. (2018)**. "UNet++: A nested U-Net architecture for medical image segmentation." *Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support*, Springer LNCS, 11045, 3-11.
- **Tags:** `[ML]` `[IMG]`
- **Context:** Enhanced U-Net with nested skip pathways
- **Key Content:**
  - Dense skip connections reduce semantic gap
  - Deep supervision improves training
  - Superior performance on medical imaging tasks
- **Implementation:** `npc_detect_initial.py` - Background estimation
- **Advantage:** More accurate background estimation than static methods
- **Cross-Reference:** See Ronneberger et al. (2015) for U-Net foundation

### Deep Learning for Background Estimation

**Weigert, M., Schmidt, U., Boothe, T., Müller, A., Dibrov, A., Jain, A., Wilhelm, B., Schmidt, D., Broaddus, C., Culley, S., Rocha-Martins, M., Segovia-Miranda, F., Norden, C., Henriques, R., Zerial, M., Solimena, M., Rink, J., Tomancak, P., Royer, L., Jug, F. & Myers, E.W. (2018)**. "Content-aware image restoration: pushing the limits of fluorescence microscopy." *Nature Methods*, 15(12), 1090-1097.
- **Tags:** `[ML]` `[MICR]` `[IMG]`
- **Context:** CARE framework for microscopy image restoration
- **Key Content:**
  - Deep learning for denoising and restoration
  - Training on paired low/high SNR images
- **Relevance:** Conceptual foundation for learned background estimation
- **Cross-Reference:** See Zhou et al. (2018) for architectural details

---

## IMAGE PROCESSING & REGISTRATION

### Phase Correlation Registration

**Reddy, B.S. & Chatterji, B.N. (1996)**. "An FFT-based technique for translation, rotation, and scale-invariant image registration." *IEEE Transactions on Image Processing*, 5(8), 1266-1271.
- **Tags:** `[IMG]` `[GEOM]`
- **Context:** Fourier-based image registration
- **Key Content:**
  - Log-polar transform for rotation and scale invariance
  - Phase correlation for translation estimation
- **Implementation:** `img_registration.py` - `compute_registration()`
- **Advantage:** Robust to noise and illumination changes
- **Cross-Reference:** See Guizar-Sicairos et al. (2008) for sub-pixel refinement

**Guizar-Sicairos, M., Thurman, S.T. & Fienup, J.R. (2008)**. "Efficient subpixel image registration algorithms." *Optics Letters*, 33(2), 156-158.
- **Tags:** `[IMG]` `[GEOM]`
- **Context:** Sub-pixel registration via matrix multiplication DFT
- **Key Content:**
  - DFT upsampling without explicit interpolation
  - Computational efficiency compared to spatial-domain methods
- **Implementation:** High-precision channel registration
- **Parameter:** `upsample_factor = 1000` for sub-pixel precision
- **Cross-Reference:** See Reddy & Chatterji (1996) for base method

### Active Contours & Snakes

**Kass, M., Witkin, A. & Terzopoulos, D. (1988)**. "Snakes: Active contour models." *International Journal of Computer Vision*, 1(4), 321-331.
- **Tags:** `[IMG]` `[GEOM]` `[OPT]`
- **Context:** Original active contour theory
- **Key Content:**
  - Energy minimization for contour fitting
  - Internal (smoothness) and external (image) energy balance
- **Implementation:** Conceptual foundation for spline fitting approach
- **Relevance:** Importance of data support for contour reliability
- **Cross-Reference:** See Sethian (1999) for level set extensions

**Sethian, J.A. (1999)**. *Level Set Methods and Fast Marching Methods*, 2nd edition. Cambridge University Press.
- **Tags:** `[IMG]` `[GEOM]`
- **Context:** Level set evolution for tracking interfaces
- **Key Content:**
  - Chapter 15: "Tracking Interfaces in Complex Flows"
  - Interpolation vs. data-driven evolution
- **Relevance:** Theoretical basis for segment reliability classification
- **Cross-Reference:** See Kass et al. (1988) for active contours

---

## SPLINE THEORY & COMPUTATIONAL GEOMETRY

### B-Spline Theory

**de Boor, C. (2001)**. *A Practical Guide to Splines*, Revised edition. Springer, New York.
- **Tags:** `[SPL]` `[GEOM]`
- **Context:** Comprehensive B-spline reference
- **Key Content:**
  - Chapters 9-14: B-spline properties and algorithms
  - Knot vector selection and parameterization
  - Numerical stability considerations
- **Implementation:** Spline fitting, refinement, and manipulation throughout pipeline
- **Requirements:** Minimum 4 control points for cubic B-splines (k=3)
- **Cross-Reference:** See Catmull & Rom (1974) for interpolating splines

**Catmull, E. & Rom, R. (1974)**. "A class of local interpolating splines." *Computer Aided Geometric Design*, Academic Press, 317-326.
- **Tags:** `[SPL]` `[GEOM]`
- **Context:** Interpolating splines for missing data
- **Key Content:**
  - Local support and continuity properties
  - Catmull-Rom splines for smooth interpolation
- **Implementation:** Foundation for Bezier bridging approach
- **Cross-Reference:** See de Boor (2001) for B-spline theory

**Farin, G. (2002)**. *Curves and Surfaces for CAGD: A Practical Guide*, 5th edition. Morgan Kaufmann.
- **Tags:** `[SPL]` `[GEOM]`
- **Context:** Computer-aided geometric design
- **Key Content:**
  - Bezier curves and spline continuity
  - Geometric properties of curves
- **Implementation:** `spline_bridging.py` - Bezier curve interpolation
- **Cross-Reference:** See de Boor (2001) and Catmull & Rom (1974)

### Computational Geometry

**Preparata, F.P. & Shamos, M.I. (1985)**. *Computational Geometry: An Introduction*. Springer-Verlag, New York.
- **Tags:** `[GEOM]`
- **Context:** Fundamental computational geometry algorithms
- **Key Content:**
  - Chapter 11: "Voronoi diagrams and nearest neighbor"
  - Point-curve distance calculations
- **Implementation:** Signed distance calculations in `ne_dual_labels.py`
- **Cross-Reference:** See O'Rourke (1998) for practical implementations

**O'Rourke, J. (1998)**. *Computational Geometry in C*, 2nd edition. Cambridge University Press.
- **Tags:** `[GEOM]`
- **Context:** Practical geometric algorithms with code
- **Key Content:**
  - Distance computation algorithms
  - Nearest-point queries on curves
- **Implementation:** Distance measurement algorithms
- **Cross-Reference:** See Preparata & Shamos (1985) for theory

**Stoker, J.J. (1969)**. *Differential Geometry*. Wiley-Interscience.
- **Tags:** `[GEOM]` `[SPL]`
- **Context:** Classical differential geometry
- **Key Content:**
  - Chapter 2: "The Local Theory of Curves"
  - Normal and tangent vectors
  - Right-hand rule for 2D curve normals
- **Implementation:** Normal vector calculation in `npc_spline_refinement.py`
- **Cross-Reference:** See Pressley (2010) for modern treatment

**Pressley, A. (2010)**. *Elementary Differential Geometry*, 2nd edition. Springer.
- **Tags:** `[GEOM]` `[SPL]`
- **Context:** Modern differential geometry textbook
- **Key Content:**
  - Section 2.2: "Arc Length and Tangent Vector"
  - Unit normal as perpendicular to unit tangent
- **Implementation:** Coordinate system management for normal directions
- **Cross-Reference:** See Stoker (1969) for classical treatment

---

## MICROSCOPY THEORY & PRACTICE

### Point Spread Function (PSF)

**Richards, B. & Wolf, E. (1959)**. "Electromagnetic diffraction in optical systems. II. Structure of the image field in an aplanatic system." *Proceedings of the Royal Society A*, 253(1274), 358-379.
- **Tags:** `[MICR]` `[IMG]`
- **Context:** Theoretical PSF for high numerical aperture objectives
- **Key Content:**
  - Vectorial diffraction theory
  - Aplanatic imaging conditions
- **Relevance:** Theoretical foundation for Richards component in Richards-Gaussian model
- **Cross-Reference:** See Zhang et al. (2007) for practical approximations

**Zhang, B., Zerubia, J. & Olivo-Marin, J.C. (2007)**. "Gaussian approximations of fluorescence microscope point-spread function models." *Applied Optics*, 46(10), 1819-1829.
- **Tags:** `[MICR]` `[IMG]`
- **Context:** PSF approximations for fluorescence microscopy
- **Key Content:**
  - Gaussian models for PSF under various conditions
  - Validation against Gibson-Lanni model
- **Implementation:** Richards-Gaussian model - Gaussian component
- **Cross-Reference:** See Gibson & Lanni (1992) for theoretical PSF

**Gibson, S.F. & Lanni, F. (1992)**. "Experimental test of an analytical model of aberration in an oil-immersion objective lens used in three-dimensional light microscopy." *Journal of the Optical Society of America A*, 9(1), 154-166.
- **Tags:** `[MICR]`
- **Context:** Theoretical PSF model including aberrations
- **Key Content:**
  - Gibson-Lanni PSF model
  - Experimental validation
- **Relevance:** Gold standard for PSF modeling
- **Cross-Reference:** See Zhang et al. (2007) for approximations

### Signal-to-Noise Ratio & Imaging

**Inoué, S. & Spring, K.R. (1997)**. *Video Microscopy: The Fundamentals*, 2nd edition. Plenum Press.
- **Tags:** `[MICR]` `[IMG]`
- **Context:** Comprehensive microscopy theory and practice
- **Key Content:**
  - Chapter 6: "Signal-to-Noise Ratio"
  - Time averaging for SNR improvement (SNR ∝ √N_frames)
- **Implementation:** Frame averaging in initial detection and registration
- **Justification:** Reduces temporal noise while preserving spatial features
- **Cross-Reference:** See Pawley (2006) for confocal/fluorescence specifics

**Pawley, J.B. (Ed.) (2006)**. *Handbook of Biological Confocal Microscopy*, 3rd edition. Springer.
- **Tags:** `[MICR]`
- **Context:** Comprehensive confocal microscopy reference
- **Key Content:**
  - Chapter 4: "Fundamental Limits in Confocal Microscopy"
  - Photon statistics and detector characteristics
- **Relevance:** Understanding fluorescence signal properties
- **Cross-Reference:** See Inoué & Spring (1997) for general microscopy

### Resolution & Localization

**Huang, B., Bates, M. & Zhuang, X. (2009)**. "Super-resolution fluorescence microscopy." *Annual Review of Biochemistry*, 78, 993-1016.
- **Tags:** `[MICR]` `[STAT]`
- **Context:** Super-resolution microscopy review
- **Key Content:**
  - Section on 2D vs 3D localization
  - Projection effects in single-plane imaging
  - PALM/STORM principles
- **Implementation:** Understanding 2D measurement limitations
- **Critical Point:** 2D distances are lower bounds on 3D separations
- **Cross-Reference:** See Hell & Wichmann (1994) for resolution limits

**Hell, S.W. & Wichmann, J. (1994)**. "Breaking the diffraction resolution limit by stimulated emission: stimulated-emission-depletion fluorescence microscopy." *Optics Letters*, 19(11), 780-782.
- **Tags:** `[MICR]`
- **Context:** STED microscopy - breaking diffraction limit
- **Key Content:**
  - Fundamental resolution limits in optical microscopy
  - Sub-diffraction imaging principles
- **Relevance:** Context for precision requirements in 2D measurements
- **Cross-Reference:** See Huang et al. (2009) for broader super-resolution context

---

## CELL BIOLOGY & BIOPHYSICS

### Nuclear Envelope & Membrane Structure

**Hetzer, M.W., Walther, T.C. & Mattaj, I.W. (2005)**. "Pushing the envelope: Structure, function, and dynamics of the nuclear periphery." *Annual Review of Cell and Developmental Biology*, 21, 347-380.
- **Tags:** `[BIO]`
- **Context:** Nuclear envelope structure and function
- **Key Content:**
  - Nuclear pore complex organization
  - Membrane curvature at pore insertion sites
- **Relevance:** Biological context for NE morphology analysis
- **Cross-Reference:** See Zimmerberg & Kozlov (2006) for membrane mechanics

**Zimmerberg, J. & Kozlov, M.M. (2006)**. "How proteins produce cellular membrane curvature." *Nature Reviews Molecular Cell Biology*, 7(1), 9-19.
- **Tags:** `[BIO]`
- **Context:** Membrane bending and curvature mechanisms
- **Key Content:**
  - Membrane bending modulus κ ≈ 10-20 kBT
  - Energy cost of curvature deformation
- **Implementation:** Curvature constraint in spline refinement
- **Justification:** Biological membranes resist extreme curvature
- **Cross-Reference:** See Hetzer et al. (2005) for NE-specific context

---

## PARAMETER DERIVATION & THRESHOLD JUSTIFICATION

### Camera Calibration Parameters

**Decision:** Camera gain, offset, read noise
**Method:** Photon transfer curve from bright/dark image pairs
**Citation:** Janesick (2001) - Chapter 3
**Implementation:** `responsivity.py`
**Rationale:** Standard method in scientific imaging; derives parameters from actual detector characteristics rather than manufacturer specifications

### Likelihood Ratio Test Threshold

**Decision:** Use AIC difference > 2 for outlier rejection
**Method:** Compare nested models (with/without point)
**Citations:** 
- Akaike (1974) - Original AIC theory
- Burnham & Anderson (2002) - Practical AIC interpretation
- Smith et al. (2010) - GLRT framework for microscopy
**Alternatives Considered:**
1. Hard-coded 5% Poisson threshold (REJECTED - arbitrary, no statistical justification)
2. Bonferroni correction (REJECTED - too conservative for dependent tests)
3. Benjamini-Hochberg FDR (REJECTED - assumes independence)
**Rationale:** AIC provides model-based comparison with built-in complexity penalty. ΔAIC > 2 is established threshold for meaningful model improvement.

### Minimum Successful Fits Threshold

**Decision:** Require ≥4 successful profile fits per segment
**Method:** Count converged optimizations per control point
**Citations:**
- de Boor (2001) - Minimum control points for cubic B-splines
- Catmull & Rom (1974) - Stability of spline interpolation
**Alternatives Considered:**
1. Use all fits regardless of count (REJECTED - unstable splines)
2. Higher threshold (e.g., 50% success rate) (CONSIDERED - may be implemented)
**Rationale:** Absolute minimum for k=3 cubic splines. Segments with fewer successful fits lack sufficient data support for reliable fitting.

### Registration Precision Threshold

**Decision:** Filter nuclei with registration error > 2σ (population-based)
**Method:** MAD-based robust standard deviation estimation
**Citations:**
- Huber & Ronchetti (2009) - Robust statistics theory
- Rousseeuw & Croux (1993) - MAD as robust scale estimator
**Alternatives Considered:**
1. Hard-coded pixel threshold (REJECTED - arbitrary, ignores experiment-specific noise)
2. Benjamini-Yekutieli FDR (CONSIDERED - for multiple comparisons)
**Rationale:** 2σ threshold is standard for outlier detection. MAD provides robustness against outliers in the threshold calculation itself. Population-based approach accounts for experiment-specific drift patterns.

### Frame Averaging Parameters

**Decision:** Average frames to improve SNR
**Method:** SNR improvement ∝ √N_frames
**Citation:** Inoué & Spring (1997) - Chapter 6
**Rationale:** Standard signal processing principle. Temporal averaging reduces noise while preserving spatial features for static/slowly moving objects like nuclear envelopes.

### Curvature Constraint

**Decision:** Limit maximum curvature during spline refinement
**Method:** Biological constraint from membrane bending energy
**Citation:** Zimmerberg & Kozlov (2006) - Membrane bending modulus
**Rationale:** Biological membranes resist extreme curvature due to bending energy costs. Constraint prevents unphysical fitting artifacts.

### 2D vs 3D Distance Measurement

**Decision:** Report as "2D lateral separation" not "3D distance"
**Method:** Single z-plane imaging provides projected distances
**Citations:**
- Huang et al. (2009) - 2D vs 3D localization
- Hell & Wichmann (1994) - Resolution limits
**Rationale:** 2D measurements are lower bounds on true 3D separations. Must acknowledge projection effects in single-plane imaging.

---

## METHODOLOGY VALIDATION CRITERIA

Each method was selected based on:

1. **Statistical Rigor:** Peer-reviewed theoretical foundation
   - All statistical tests have published theoretical basis
   - Parameters derived from first principles when possible

2. **Domain Appropriateness:** Specifically applicable to fluorescence microscopy
   - Methods validated on biological imaging data
   - Account for photon statistics and detector characteristics

3. **Computational Tractability:** Implementable with reasonable resources
   - Algorithms scale to typical dataset sizes
   - Balance between accuracy and computational cost

4. **Empirical Validation:** Tested on actual biological data
   - Methods work on real NE images with typical SNR
   - Robustness verified across multiple experiments

---

## CROSS-REFERENCE INDEX

### By Research Question

**"How do we estimate camera parameters?"**
→ Janesick (2001), Gonzalez & Woods (2008)

**"How do we detect outliers in profile fits?"**
→ Smith et al. (2010), Akaike (1974), Burnham & Anderson (2002)

**"How do we optimize spline control points?"**
→ Marquardt (1963), Dennis & Schnabel (1996), Bevington & Robinson (2003)

**"How do we handle gaps in spline data?"**
→ de Boor (2001), Catmull & Rom (1974), Farin (2002)

**"How do we register multi-channel images?"**
→ Reddy & Chatterji (1996), Guizar-Sicairos et al. (2008)

**"What are biological constraints on membrane shape?"**
→ Zimmerberg & Kozlov (2006), Hetzer et al. (2005)

### By Pipeline Component

**Camera Calibration:**
→ Janesick (2001), Gonzalez & Woods (2008)

**Initial NE Detection:**
→ Zhou et al. (2018), Ronneberger et al. (2015), Weigert et al. (2018)

**Image Registration:**
→ Reddy & Chatterji (1996), Guizar-Sicairos et al. (2008)

**Registration Filtering:**
→ Rousseeuw & Croux (1993), Huber & Ronchetti (2009), Benjamini & Yekutieli (2001)

**Spline Refinement:**
→ Marquardt (1963), Dennis & Schnabel (1996), Smith et al. (2010), de Boor (2001)

**Spline Bridging:**
→ Catmull & Rom (1974), Farin (2002), Kass et al. (1988)

**Distance Calculation:**
→ Preparata & Shamos (1985), O'Rourke (1998), Huang et al. (2009)


---

*This bibliography is actively maintained and updated as new methods are implemented or citations are required for manuscript preparation.*
