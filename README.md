# SimpleITK documentation

This file was auto-generated using a Python script


### sitk.Abs
    Abs(Image image1) -> Image



    Computes the absolute value of each pixel.


    This function directly calls the execute method of AbsImageFilter in order to support a procedural API


    See:
     itk::simple::AbsImageFilter for the object oriented interface



    
### sitk.AbsImageFilter


    Computes the absolute value of each pixel.


    itk::Math::abs() is used to perform the computation.
    See:
     itk::simple::Abs for the procedural interface

     itk::AbsImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkAbsImageFilter.h

    
### sitk.AbsoluteValueDifference
    AbsoluteValueDifference(Image image1, Image image2) -> Image
    AbsoluteValueDifference(Image image1, double constant) -> Image
    AbsoluteValueDifference(double constant, Image image2) -> Image



    
### sitk.AbsoluteValueDifferenceImageFilter


    Implements pixel-wise the computation of absolute value difference.


    This filter is parametrized over the types of the two input images and
    the type of the output image.

    Numeric conversions (castings) are done by the C++ defaults.

    The filter will walk over all the pixels in the two input images, and
    for each one of them it will do the following:


    Cast the input 1 pixel value to double .

    Cast the input 2 pixel value to double .

    Compute the difference of the two pixel values.

    Compute the absolute value of the difference.

    Cast the double value resulting from the absolute value to the pixel
    type of the output image.

    Store the casted value into the output image.
     The filter expects all images to have the same dimension (e.g. all
    2D, or all 3D, or all ND).
    See:
     itk::simple::AbsoluteValueDifference for the procedural interface

     itk::AbsoluteValueDifferenceImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkAbsoluteValueDifferenceImageFilter.h

    
### sitk.Acos
    Acos(Image image1) -> Image



    Computes the inverse cosine of each pixel.


    This function directly calls the execute method of AcosImageFilter in order to support a procedural API


    See:
     itk::simple::AcosImageFilter for the object oriented interface



    
### sitk.AcosImageFilter


    Computes the inverse cosine of each pixel.


    This filter is templated over the pixel type of the input image and
    the pixel type of the output image.

    The filter walks over all the pixels in the input image, and for each
    pixel does do the following:


    cast the pixel value to double ,

    apply the std::acos() function to the double value

    cast the double value resulting from std::acos() to the pixel type of
    the output image

    store the casted value into the output image.
     The filter expects both images to have the same dimension (e.g. both
    2D, or both 3D, or both ND).
    See:
     itk::simple::Acos for the procedural interface

     itk::AcosImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkAcosImageFilter.h

    
### sitk.AdaptiveHistogramEqualization
    AdaptiveHistogramEqualization(Image image1, VectorUInt32 radius, float alpha=0.3, float beta=0.3) -> Image



    Power Law Adaptive Histogram Equalization.


    This function directly calls the execute method of AdaptiveHistogramEqualizationImageFilter in order to support a procedural API


    See:
     itk::simple::AdaptiveHistogramEqualizationImageFilter for the object oriented interface



    
### sitk.AdaptiveHistogramEqualizationImageFilter


    Power Law Adaptive Histogram Equalization.


    Histogram equalization modifies the contrast in an image. The AdaptiveHistogramEqualizationImageFilter is a superset of many contrast enhancing filters. By modifying its
    parameters (alpha, beta, and window), the AdaptiveHistogramEqualizationImageFilter can produce an adaptively equalized histogram or a version of unsharp
    mask (local mean subtraction). Instead of applying a strict histogram
    equalization in a window about a pixel, this filter prescribes a
    mapping function (power law) controlled by the parameters alpha and
    beta.

    The parameter alpha controls how much the filter acts like the
    classical histogram equalization method (alpha=0) to how much the
    filter acts like an unsharp mask (alpha=1).

    The parameter beta controls how much the filter acts like an unsharp
    mask (beta=0) to much the filter acts like pass through (beta=1, with
    alpha=1).

    The parameter window controls the size of the region over which local
    statistics are calculated.

    By altering alpha, beta and window, a host of equalization and unsharp
    masking filters is available.

    The boundary condition ignores the part of the neighborhood outside
    the image, and over-weights the valid part of the neighborhood.

    For detail description, reference "Adaptive Image Contrast
    Enhancement using Generalizations of Histogram Equalization." J.Alex
    Stark. IEEE Transactions on Image Processing, May 2000.
    See:
     itk::simple::AdaptiveHistogramEqualization for the procedural interface

     itk::AdaptiveHistogramEqualizationImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkAdaptiveHistogramEqualizationImageFilter.h

    
### sitk.Add
    Add(Image image1, Image image2) -> Image
    Add(Image image1, double constant) -> Image
    Add(double constant, Image image2) -> Image



    
### sitk.AddImageFilter


    Pixel-wise addition of two images.


    This class is templated over the types of the two input images and the
    type of the output image. Numeric conversions (castings) are done by
    the C++ defaults.

    The pixel type of the input 1 image must have a valid definition of
    the operator+ with a pixel type of the image 2. This condition is
    required because internally this filter will perform the operation


    Additionally the type resulting from the sum, will be cast to the
    pixel type of the output image.

    The total operation over one pixel will be

    For example, this filter could be used directly for adding images
    whose pixels are vectors of the same dimension, and to store the
    resulting vector in an output image of vector pixels.

    The images to be added are set using the methods:

    Additionally, this filter can be used to add a constant to every pixel
    of an image by using


    WARNING:
    No numeric overflow checking is performed in this filter.

    See:
     itk::simple::Add for the procedural interface

     itk::AddImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkAddImageFilter.h

    
### sitk.AdditiveGaussianNoise
    AdditiveGaussianNoise(Image image1, double standardDeviation=1.0, double mean=0.0, uint32_t seed) -> Image



    Alter an image with additive Gaussian white noise.


    This function directly calls the execute method of AdditiveGaussianNoiseImageFilter in order to support a procedural API


    See:
     itk::simple::AdditiveGaussianNoiseImageFilter for the object oriented interface



    
### sitk.AdditiveGaussianNoiseImageFilter


    Alter an image with additive Gaussian white noise.


    Additive Gaussian white noise can be modeled as:


    $ I = I_0 + N $

    where $ I $ is the observed image, $ I_0 $ is the noise-free image and $ N $ is a normally distributed random variable of mean $ \mu $ and variance $ \sigma^2 $ :

    $ N \sim \mathcal{N}(\mu, \sigma^2) $
     The noise is independent of the pixel intensities.


    Gaetan Lehmann
     This code was contributed in the Insight Journal paper "Noise
    Simulation". https://hdl.handle.net/10380/3158
    See:
     itk::simple::AdditiveGaussianNoise for the procedural interface

     itk::AdditiveGaussianNoiseImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkAdditiveGaussianNoiseImageFilter.h

    
### sitk.AffineTransform


    An affine transformation about a fixed center with translation for a
    2D or 3D coordinate.



    See:
     itk::AffineTransform


    C++ includes: sitkAffineTransform.h

    
### sitk.AggregateLabelMap
    AggregateLabelMap(Image image1) -> Image



    Collapses all labels into the first label.


    This function directly calls the execute method of AggregateLabelMapFilter in order to support a procedural API


    See:
     itk::simple::AggregateLabelMapFilter for the object oriented interface



    
### sitk.AggregateLabelMapFilter


    Collapses all labels into the first label.


    This filter takes a label map as input and visits the pixels of all
    labels and assigns them to the first label of the label map. At the
    end of the execution of this filter, the map will contain a single
    filter.

    This implementation was taken from the Insight Journal paper: https://hdl.handle.net/1926/584 or http://www.insight-journal.org/browse/publication/176


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.

    See:
     ShapeLabelObject , RelabelComponentImageFilter

     itk::simple::AggregateLabelMapFilter for the procedural interface

     itk::AggregateLabelMapFilter for the Doxygen on the original ITK class.


    C++ includes: sitkAggregateLabelMapFilter.h

    
### sitk.And
    And(Image image1, Image image2) -> Image
    And(Image image1, int constant) -> Image
    And(int constant, Image image2) -> Image



    
### sitk.AndImageFilter


    Implements the AND bitwise operator pixel-wise between two images.


    This class is templated over the types of the two input images and the
    type of the output image. Numeric conversions (castings) are done by
    the C++ defaults.

    Since the bitwise AND operation is only defined in C++ for integer
    types, the images passed to this filter must comply with the
    requirement of using integer pixel type.

    The total operation over one pixel will be Where "&" is the bitwise AND operator in C++.
    See:
     itk::simple::And for the procedural interface

     itk::AndImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkAndImageFilter.h

    
### sitk.AntiAliasBinary
    AntiAliasBinary(Image image1, double maximumRMSError=0.07, uint32_t numberOfIterations=1000) -> Image



    A method for estimation of a surface from a binary volume.


    This function directly calls the execute method of AntiAliasBinaryImageFilter in order to support a procedural API


    See:
     itk::simple::AntiAliasBinaryImageFilter for the object oriented interface



    
### sitk.AntiAliasBinaryImageFilter


    A method for estimation of a surface from a binary volume.



    This filter implements a surface-fitting method for estimation of a
    surface from a binary volume. This process can be used to reduce
    aliasing artifacts which result in visualization of binary partitioned
    surfaces.

    The binary volume (filter input) is used as a set of constraints in an
    iterative relaxation process of an estimated ND surface. The surface
    is described implicitly as the zero level set of a volume $ \phi $ and allowed to deform under curvature flow. A set of constraints is
    imposed on this movement as follows:

    \[ u_{i,j,k}^{n+1} = \left\{ \begin{array}{ll}
    \mbox{max} (u_{i,j,k}^{n} + \Delta t H_{i,j,k}^{n}, 0) &
    \mbox{\f$B_{i,j,k} = 1\f$} \\ \mbox{min}
    (u_{i,j,k}^{n} + \Delta t H_{i,j,k}^{n}, 0) &
    \mbox{\f$B_{i,j,k} = -1\f$} \end{array}\right. \]

    where $ u_{i,j,k}^{n} $ is the value of $ \phi $ at discrete index $ (i,j,k) $ and iteration $ n $ , $ H $ is the gradient magnitude times mean curvature of $ \phi $ , and $ B $ is the binary input volume, with 1 denoting an inside pixel and -1
    denoting an outside pixel.
    NOTES
    This implementation uses a sparse field level set solver instead of
    the narrow band implementation described in the reference below, which
    may introduce some differences in how fast and how accurately (in
    terms of RMS error) the solution converges.
    REFERENCES
    Whitaker, Ross. "Reducing Aliasing Artifacts In Iso-Surfaces of
    Binary Volumes" IEEE Volume Visualization and Graphics Symposium,
    October 2000, pp.23-32.
    PARAMETERS
    The MaximumRMSChange parameter is used to determine when the solution
    has converged. A lower value will result in a tighter-fitting
    solution, but will require more computations. Too low a value could
    put the solver into an infinite loop. Values should always be less
    than 1.0. A value of 0.07 is a good starting estimate.

    The MaximumIterations parameter can be used to halt the solution after
    a specified number of iterations.
    INPUT
    The input is an N-dimensional image of any type. It is assumed to be a
    binary image. The filter will use an isosurface value that is halfway
    between the min and max values in the image. A signed data type is not
    necessary for the input.
    OUTPUT
    The filter will output a level set image of real, signed values. The
    zero crossings of this (N-dimensional) image represent the position of
    the isosurface value of interest. Values outside the zero level set
    are negative and values inside the zero level set are positive values.
    IMPORTANT!
    The output image type you use to instantiate this filter should be a
    real valued scalar type. In other words: doubles or floats.
    USING THIS FILTER
    The filter is relatively straightforward to use. Tests and examples
    exist to illustrate. The important thing is to understand the input
    and output types so you can properly interperet your results.

    In the common case, the only parameter that will need to be set is the
    MaximumRMSChange parameter, which determines when the solver halts.

    See:
     itk::simple::AntiAliasBinary for the procedural interface

     itk::AntiAliasBinaryImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkAntiAliasBinaryImageFilter.h

    
### sitk.ApproximateSignedDistanceMap
    ApproximateSignedDistanceMap(Image image1, double insideValue=1, double outsideValue=0) -> Image



    Create a map of the approximate signed distance from the boundaries of
    a binary image.


    This function directly calls the execute method of ApproximateSignedDistanceMapImageFilter in order to support a procedural API


    See:
     itk::simple::ApproximateSignedDistanceMapImageFilter for the object oriented interface



    
### sitk.ApproximateSignedDistanceMapImageFilter


    Create a map of the approximate signed distance from the boundaries of
    a binary image.


    The ApproximateSignedDistanceMapImageFilter takes as input a binary image and produces a signed distance map.
    Each pixel value in the output contains the approximate distance from
    that pixel to the nearest "object" in the binary image. This filter
    differs from the DanielssonDistanceMapImageFilter in that it calculates the distance to the "object edge" for pixels
    within the object.

    Negative values in the output indicate that the pixel at that position
    is within an object in the input image. The absolute value of a
    negative pixel represents the approximate distance to the nearest
    object boundary pixel.

    WARNING: This filter requires that the output type be floating-point.
    Otherwise internal calculations will not be performed to the
    appropriate precision, resulting in completely incorrect (read: zero-
    valued) output.

    The distances computed by this filter are Chamfer distances, which are
    only an approximation to Euclidian distances, and are not as exact
    approximations as those calculated by the DanielssonDistanceMapImageFilter . On the other hand, this filter is faster.

    This filter requires that an "inside value" and "outside value" be
    set as parameters. The "inside value" is the intensity value of the
    binary image which corresponds to objects, and the "outside value"
    is the intensity of the background. (A typical binary image often
    represents objects as black (0) and background as white (usually 255),
    or vice-versa.) Note that this filter is slightly faster if the inside
    value is less than the outside value. Otherwise an extra iteration
    through the image is required.

    This filter uses the FastChamferDistanceImageFilter and the IsoContourDistanceImageFilter internally to perform the distance calculations.


    See:
     DanielssonDistanceMapImageFilter

     SignedDanielssonDistanceMapImageFilter

     SignedMaurerDistanceMapImageFilter

     FastChamferDistanceImageFilter

     IsoContourDistanceImageFilter

    Zach Pincus

    See:
     itk::simple::ApproximateSignedDistanceMap for the procedural interface

     itk::ApproximateSignedDistanceMapImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkApproximateSignedDistanceMapImageFilter.h

    
### sitk.Asin
    Asin(Image image1) -> Image



    Computes the sine of each pixel.


    This function directly calls the execute method of AsinImageFilter in order to support a procedural API


    See:
     itk::simple::AsinImageFilter for the object oriented interface



    
### sitk.AsinImageFilter


    Computes the sine of each pixel.


    This filter is templated over the pixel type of the input image and
    the pixel type of the output image.

    The filter walks over all the pixels in the input image, and for each
    pixel does the following:


    cast the pixel value to double ,

    apply the std::asin() function to the double value,

    cast the double value resulting from std::asin() to the pixel type of
    the output image,

    store the casted value into the output image.
     The filter expects both images to have the same dimension (e.g. both
    2D, or both 3D, or both ND)
    See:
     itk::simple::Asin for the procedural interface

     itk::AsinImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkAsinImageFilter.h

    
### sitk.Atan
    Atan(Image image1) -> Image



    Computes the one-argument inverse tangent of each pixel.


    This function directly calls the execute method of AtanImageFilter in order to support a procedural API


    See:
     itk::simple::AtanImageFilter for the object oriented interface



    
### sitk.Atan2
    Atan2(Image image1, Image image2) -> Image
    Atan2(Image image1, double constant) -> Image
    Atan2(double constant, Image image2) -> Image



    
### sitk.Atan2ImageFilter


    Computes two argument inverse tangent.


    The first argument to the atan function is provided by a pixel in the
    first input image (SetInput1() ) and the corresponding pixel in the
    second input image (SetInput2() ) is used as the second argument.

    This class is templated over the types of the two input images and the
    type of the output image. Numeric conversions (castings) are done by
    the C++ defaults.

    Both pixel input types are cast to double in order to be used as
    parameters of std::atan2() . The resulting double value is cast to the
    output pixel type.
    See:
     itk::simple::Atan2 for the procedural interface

     itk::Atan2ImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkAtan2ImageFilter.h

    
### sitk.AtanImageFilter


    Computes the one-argument inverse tangent of each pixel.


    This filter is templated over the pixel type of the input image and
    the pixel type of the output image.

    The filter walks over all the pixels in the input image, and for each
    pixel does the following:


    cast the pixel value to double ,

    apply the std::atan() function to the double value,

    cast the double value resulting from std::atan() to the pixel type of
    the output image,

    store the cast value into the output image.
    See:
     itk::simple::Atan for the procedural interface

     itk::AtanImageFilter for the Doxygen on the original ITK class.



    C++ includes: sitkAtanImageFilter.h

    
### sitk.BSplineDecomposition
    BSplineDecomposition(Image image1, uint32_t splineOrder=3) -> Image



    Calculates the B-Spline coefficients of an image. Spline order may be
    from 0 to 5.


    This function directly calls the execute method of BSplineDecompositionImageFilter in order to support a procedural API


    See:
     itk::simple::BSplineDecompositionImageFilter for the object oriented interface



    
### sitk.BSplineDecompositionImageFilter


    Calculates the B-Spline coefficients of an image. Spline order may be
    from 0 to 5.


    This class defines N-Dimension B-Spline transformation. It is based
    on: [1] M. Unser, "Splines: A Perfect Fit for Signal and Image
    Processing," IEEE Signal Processing Magazine, vol. 16, no. 6, pp.
    22-38, November 1999. [2] M. Unser, A. Aldroubi and M. Eden,
    "B-Spline Signal Processing: Part I--Theory," IEEE Transactions on
    Signal Processing, vol. 41, no. 2, pp. 821-832, February 1993. [3] M.
    Unser, A. Aldroubi and M. Eden, "B-Spline Signal Processing: Part II
    --Efficient Design and Applications," IEEE Transactions on Signal
    Processing, vol. 41, no. 2, pp. 834-848, February 1993. And code obtained from bigwww.epfl.ch by Philippe Thevenaz

    Limitations: Spline order must be between 0 and 5. Spline order must
    be set before setting the image. Uses mirror boundary conditions.
    Requires the same order of Spline for each dimension. Can only process
    LargestPossibleRegion


    See:
     itkBSplineInterpolateImageFunction

     itk::simple::BSplineDecomposition for the procedural interface

     itk::BSplineDecompositionImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkBSplineDecompositionImageFilter.h

    
### sitk.BSplineTransform


    A deformable transform over a bounded spatial domain using a BSpline
    representation for a 2D or 3D coordinate space.



    See:
     itk::BSplineTransform


    C++ includes: sitkBSplineTransform.h

    
### sitk.BSplineTransformInitializer
    BSplineTransformInitializer(Image image1, VectorUInt32 transformDomainMeshSize, unsigned int order=3) -> BSplineTransform



    BSplineTransformInitializerFilter is a helper class intended to initialize the control point grid such
    that it has a physically consistent definition. It sets the transform
    domain origin, physical dimensions and direction from information
    obtained from the image. It also sets the mesh size if asked to do so
    by calling SetTransformDomainMeshSize()before calling
    InitializeTransform().


    This function directly calls the execute method of BSplineTransformInitializerFilter in order to support a procedural API


    See:
     itk::simple::BSplineTransformInitializerFilter for the object oriented interface



    
### sitk.BSplineTransformInitializerFilter


    BSplineTransformInitializerFilter is a helper class intended to initialize the control point grid such
    that it has a physically consistent definition. It sets the transform
    domain origin, physical dimensions and direction from information
    obtained from the image. It also sets the mesh size if asked to do so
    by calling SetTransformDomainMeshSize()before calling InitializeTransform().



    Luis Ibanez Nick Tustison

    See:
     itk::simple::BSplineTransformInitializer for the procedural interface

     itk::BSplineTransformInitializer for the Doxygen on the original ITK class.


    C++ includes: sitkBSplineTransformInitializerFilter.h

    
### sitk.Bilateral
    Bilateral(Image image1, double domainSigma=4.0, double rangeSigma=50.0, unsigned int numberOfRangeGaussianSamples=100) -> Image



    Blurs an image while preserving edges.


    This function directly calls the execute method of BilateralImageFilter in order to support a procedural API


    See:
     itk::simple::BilateralImageFilter for the object oriented interface



    
### sitk.BilateralImageFilter


    Blurs an image while preserving edges.


    This filter uses bilateral filtering to blur an image using both
    domain and range "neighborhoods". Pixels that are close to a pixel
    in the image domain and similar to a pixel in the image range are used
    to calculate the filtered value. Two gaussian kernels (one in the
    image domain and one in the image range) are used to smooth the image.
    The result is an image that is smoothed in homogeneous regions yet has
    edges preserved. The result is similar to anisotropic diffusion but
    the implementation in non-iterative. Another benefit to bilateral
    filtering is that any distance metric can be used for kernel smoothing
    the image range. Hence, color images can be smoothed as vector images,
    using the CIE distances between intensity values as the similarity
    metric (the Gaussian kernel for the image domain is evaluated using
    CIE distances). A separate version of this filter will be designed for
    color and vector images.

    Bilateral filtering is capable of reducing the noise in an image by an
    order of magnitude while maintaining edges.

    The bilateral operator used here was described by Tomasi and Manduchi
    (Bilateral Filtering for Gray and ColorImages. IEEE ICCV. 1998.)


    See:
     GaussianOperator

     RecursiveGaussianImageFilter

     DiscreteGaussianImageFilter

     AnisotropicDiffusionImageFilter

     Image

     Neighborhood

     NeighborhoodOperator
     TodoSupport color images

    Support vector images
    See:
     itk::simple::Bilateral for the procedural interface

     itk::BilateralImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkBilateralImageFilter.h

    
### sitk.BinShrink
    BinShrink(Image image1, VectorUInt32 shrinkFactors) -> Image



    Reduce the size of an image by an integer factor in each dimension
    while performing averaging of an input neighborhood.


    This function directly calls the execute method of BinShrinkImageFilter in order to support a procedural API


    See:
     itk::simple::BinShrinkImageFilter for the object oriented interface



    
### sitk.BinShrinkImageFilter


    Reduce the size of an image by an integer factor in each dimension
    while performing averaging of an input neighborhood.


    The output image size in each dimension is given by:

    outputSize[j] = max( std::floor(inputSize[j]/shrinkFactor[j]), 1 );

    The algorithm implemented can be describe with the following equation
    for 2D: \[ \mathsf{I}_{out}(x_o,x_1) =
    \frac{\sum_{i=0}^{f_0}\sum_{j=0}^{f_1}\mathsf{I}_{in}(f_0
    x_o+i,f_1 x_1+j)}{f_0 f_1} \]

    This filter is implemented so that the starting extent of the first
    pixel of the output matches that of the input.

    The change in image geometry from a 5x5 image binned by a factor of
    2x2. This code was contributed in the Insight Journal paper:
    "BinShrink: A multi-resolution filter with cache efficient
    averaging" by Lowekamp B., Chen D. https://hdl.handle.net/10380/3450
    See:
     itk::simple::BinShrink for the procedural interface

     itk::BinShrinkImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkBinShrinkImageFilter.h

    
### sitk.BinaryClosingByReconstruction
    BinaryClosingByReconstruction(Image arg1, uint32_t radius=1, itk::simple::KernelEnum kernel, double foregroundValue=1.0, bool fullyConnected=False) -> Image
    BinaryClosingByReconstruction(Image arg1, VectorUInt32 vectorRadius, itk::simple::KernelEnum kernel, double foregroundValue=1.0, bool fullyConnected=False) -> Image



    itk::simple::BinaryClosingByReconstructionImageFilter Functional Interface

    This function directly calls the execute method of BinaryClosingByReconstructionImageFilter in order to support a fully functional API


    
### sitk.BinaryClosingByReconstructionImageFilter


    binary closing by reconstruction of an image.


    This filter removes small (i.e., smaller than the structuring element)
    holes in the image. It is defined as: Closing(f) =
    ReconstructionByErosion(Dilation(f)).

    The structuring element is assumed to be composed of binary values
    (zero or one). Only elements of the structuring element having values
    > 0 are candidates for affecting the center pixel.


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.
     This implementation was taken from the Insight Journal paper: https://hdl.handle.net/1926/584 or http://www.insight-journal.org/browse/publication/176


    See:
     MorphologyImageFilter , ClosingByReconstructionImageFilter , BinaryOpeningByReconstructionImageFilter

     itk::simple::BinaryClosingByReconstruction for the procedural interface

     itk::BinaryClosingByReconstructionImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkBinaryClosingByReconstructionImageFilter.h

    
### sitk.BinaryContour
    BinaryContour(Image image1, bool fullyConnected=False, double backgroundValue=0.0, double foregroundValue=1.0) -> Image



    Labels the pixels on the border of the objects in a binary image.


    This function directly calls the execute method of BinaryContourImageFilter in order to support a procedural API


    See:
     itk::simple::BinaryContourImageFilter for the object oriented interface



    
### sitk.BinaryContourImageFilter


    Labels the pixels on the border of the objects in a binary image.


    BinaryContourImageFilter takes a binary image as input, where the pixels in the objects are
    the pixels with a value equal to ForegroundValue. Only the pixels on
    the contours of the objects are kept. The pixels not on the border are
    changed to BackgroundValue.

    The connectivity can be changed to minimum or maximum connectivity
    with SetFullyConnected() . Full connectivity produces thicker contours.

    https://hdl.handle.net/1926/1352


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.

    See:
     LabelContourImageFilter BinaryErodeImageFilter SimpleContourExtractorImageFilter

     itk::simple::BinaryContour for the procedural interface

     itk::BinaryContourImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkBinaryContourImageFilter.h

    
### sitk.BinaryDilate
    BinaryDilate(Image arg1, uint32_t radius=1, itk::simple::KernelEnum kernel, double backgroundValue=0.0, double foregroundValue=1.0, bool boundaryToForeground=False) -> Image
    BinaryDilate(Image arg1, VectorUInt32 vectorRadius, itk::simple::KernelEnum kernel, double backgroundValue=0.0, double foregroundValue=1.0, bool boundaryToForeground=False) -> Image



    itk::simple::BinaryDilateImageFilter Functional Interface

    This function directly calls the execute method of BinaryDilateImageFilter in order to support a fully functional API


    
### sitk.BinaryDilateImageFilter


    Fast binary dilation.


    BinaryDilateImageFilter is a binary dilation morphologic operation. This implementation is
    based on the papers:

    L.Vincent "Morphological transformations of binary images with
    arbitrary structuring elements", and

    N.Nikopoulos et al. "An efficient algorithm for 3d binary
    morphological transformations with 3d structuring elements for
    arbitrary size and shape". IEEE Transactions on Image Processing. Vol. 9. No. 3. 2000. pp. 283-286.

    Gray scale images can be processed as binary images by selecting a
    "DilateValue". Pixel values matching the dilate value are considered
    the "foreground" and all other pixels are "background". This is
    useful in processing segmented images where all pixels in segment #1
    have value 1 and pixels in segment #2 have value 2, etc. A particular
    "segment number" can be processed. DilateValue defaults to the
    maximum possible value of the PixelType.

    The structuring element is assumed to be composed of binary values
    (zero or one). Only elements of the structuring element having values
    > 0 are candidates for affecting the center pixel. A reasonable choice
    of structuring element is itk::BinaryBallStructuringElement .


    See:
     ImageToImageFilter BinaryErodeImageFilter BinaryMorphologyImageFilter

     itk::simple::BinaryDilate for the procedural interface

     itk::BinaryDilateImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkBinaryDilateImageFilter.h

    
### sitk.BinaryErode
    BinaryErode(Image arg1, uint32_t radius=1, itk::simple::KernelEnum kernel, double backgroundValue=0.0, double foregroundValue=1.0, bool boundaryToForeground=True) -> Image
    BinaryErode(Image arg1, VectorUInt32 vectorRadius, itk::simple::KernelEnum kernel, double backgroundValue=0.0, double foregroundValue=1.0, bool boundaryToForeground=True) -> Image



    itk::simple::BinaryErodeImageFilter Functional Interface

    This function directly calls the execute method of BinaryErodeImageFilter in order to support a fully functional API


    
### sitk.BinaryErodeImageFilter


    Fast binary erosion.


    BinaryErodeImageFilter is a binary erosion morphologic operation. This implementation is
    based on the papers:

    L.Vincent "Morphological transformations of binary images with
    arbitrary structuring elements", and

    N.Nikopoulos et al. "An efficient algorithm for 3d binary
    morphological transformations with 3d structuring elements for
    arbitrary size and shape". IEEE Transactions on Image Processing. Vol. 9. No. 3. 2000. pp. 283-286.

    Gray scale images can be processed as binary images by selecting a
    "ErodeValue". Pixel values matching the erode value are considered
    the "foreground" and all other pixels are "background". This is
    useful in processing segmented images where all pixels in segment #1
    have value 1 and pixels in segment #2 have value 2, etc. A particular
    "segment number" can be processed. ErodeValue defaults to the
    maximum possible value of the PixelType. The eroded pixels will
    receive the BackgroundValue (defaults to 0).

    The structuring element is assumed to be composed of binary values
    (zero or one). Only elements of the structuring element having values
    > 0 are candidates for affecting the center pixel. A reasonable choice
    of structuring element is itk::BinaryBallStructuringElement .


    See:
     ImageToImageFilter BinaryDilateImageFilter BinaryMorphologyImageFilter

     itk::simple::BinaryErode for the procedural interface

     itk::BinaryErodeImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkBinaryErodeImageFilter.h

    
### sitk.BinaryFillhole
    BinaryFillhole(Image image1, bool fullyConnected=False, double foregroundValue=1.0) -> Image



    Remove holes not connected to the boundary of the image.


    This function directly calls the execute method of BinaryFillholeImageFilter in order to support a procedural API


    See:
     itk::simple::BinaryFillholeImageFilter for the object oriented interface



    
### sitk.BinaryFillholeImageFilter


    Remove holes not connected to the boundary of the image.


    BinaryFillholeImageFilter fills holes in a binary image.

    Geodesic morphology and the Fillhole algorithm is described in Chapter
    6 of Pierre Soille's book "Morphological Image Analysis: Principles
    and Applications", Second Edition, Springer, 2003.


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.
     This implementation was taken from the Insight Journal paper: https://hdl.handle.net/1926/584 or http://www.insight-journal.org/browse/publication/176


    See:
     GrayscaleFillholeImageFilter

     itk::simple::BinaryFillhole for the procedural interface

     itk::BinaryFillholeImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkBinaryFillholeImageFilter.h

    
### sitk.BinaryGrindPeak
    BinaryGrindPeak(Image image1, bool fullyConnected=False, double foregroundValue=1.0, double backgroundValue=0) -> Image



    Remove the objects not connected to the boundary of the image.


    This function directly calls the execute method of BinaryGrindPeakImageFilter in order to support a procedural API


    See:
     itk::simple::BinaryGrindPeakImageFilter for the object oriented interface



    
### sitk.BinaryGrindPeakImageFilter


    Remove the objects not connected to the boundary of the image.


    BinaryGrindPeakImageFilter ginds peaks in a grayscale image.

    Geodesic morphology and the grind peak algorithm is described in
    Chapter 6 of Pierre Soille's book "Morphological Image Analysis:
    Principles and Applications", Second Edition, Springer, 2003.


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.
     This implementation was taken from the Insight Journal paper: https://hdl.handle.net/1926/584 or http://www.insight-journal.org/browse/publication/176


    See:
     GrayscaleGrindPeakImageFilter

     itk::simple::BinaryGrindPeak for the procedural interface

     itk::BinaryGrindPeakImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkBinaryGrindPeakImageFilter.h

    
### sitk.BinaryImageToLabelMap
    BinaryImageToLabelMap(Image image1, bool fullyConnected=False, double inputForegroundValue=1.0, double outputBackgroundValue=0.0) -> Image



    Label the connected components in a binary image and produce a
    collection of label objects.


    This function directly calls the execute method of BinaryImageToLabelMapFilter in order to support a procedural API


    See:
     itk::simple::BinaryImageToLabelMapFilter for the object oriented interface



    
### sitk.BinaryImageToLabelMapFilter


    Label the connected components in a binary image and produce a
    collection of label objects.


    BinaryImageToLabelMapFilter labels the objects in a binary image. Each distinct object is
    assigned a unique label. The final object labels start with 1 and are
    consecutive. Objects that are reached earlier by a raster order scan
    have a lower label.

    The GetOutput() function of this class returns an itk::LabelMap .

    This implementation was taken from the Insight Journal paper: https://hdl.handle.net/1926/584 or http://www.insight-journal.org/browse/publication/176


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.

    See:
     ConnectedComponentImageFilter , LabelImageToLabelMapFilter , LabelMap , LabelObject

     itk::simple::BinaryImageToLabelMapFilter for the procedural interface

     itk::BinaryImageToLabelMapFilter for the Doxygen on the original ITK class.


    C++ includes: sitkBinaryImageToLabelMapFilter.h

    
### sitk.BinaryMagnitude
    BinaryMagnitude(Image image1, Image image2) -> Image



    Computes the square root of the sum of squares of corresponding input
    pixels.


    This function directly calls the execute method of BinaryMagnitudeImageFilter in order to support a procedural API


    See:
     itk::simple::BinaryMagnitudeImageFilter for the object oriented interface



    
### sitk.BinaryMagnitudeImageFilter


    Computes the square root of the sum of squares of corresponding input
    pixels.


    This filter is templated over the types of the two input images and
    the type of the output image.

    Numeric conversions (castings) are done by the C++ defaults.

    The filter walks over all of the pixels in the two input images, and
    for each pixel does the following:


    cast the input 1 pixel value to double

    cast the input 2 pixel value to double

    compute the sum of squares of the two pixel values

    compute the square root of the sum

    cast the double value resulting from std::sqrt() to the pixel type of
    the output image

    store the cast value into the output image.
     The filter expects all images to have the same dimension (e.g. all
    2D, or all 3D, or all ND)
    See:
     itk::simple::BinaryMagnitude for the procedural interface

     itk::BinaryMagnitudeImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkBinaryMagnitudeImageFilter.h

    
### sitk.BinaryMedian
    BinaryMedian(Image image1, VectorUInt32 radius, double foregroundValue=1.0, double backgroundValue=0.0) -> Image



    Applies a version of the median filter optimized for binary images.


    This function directly calls the execute method of BinaryMedianImageFilter in order to support a procedural API


    See:
     itk::simple::BinaryMedianImageFilter for the object oriented interface



    
### sitk.BinaryMedianImageFilter


    Applies a version of the median filter optimized for binary images.


    This filter was contributed by Bjorn Hanch Sollie after identifying
    that the generic Median filter performed unnecessary operations when
    the input image is binary.

    This filter computes an image where a given pixel is the median value
    of the pixels in a neighborhood about the corresponding input pixel.
    For the case of binary images the median can be obtained by simply
    counting the neighbors that are foreground.

    A median filter is one of the family of nonlinear filters. It is used
    to smooth an image without being biased by outliers or shot noise.


    See:
     Image

     Neighborhood

     NeighborhoodOperator

     NeighborhoodIterator

     itk::simple::BinaryMedian for the procedural interface

     itk::BinaryMedianImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkBinaryMedianImageFilter.h

    
### sitk.BinaryMinMaxCurvatureFlow
    BinaryMinMaxCurvatureFlow(Image image1, double timeStep=0.05, uint32_t numberOfIterations=5, int stencilRadius=2, double threshold=0) -> Image



    Denoise a binary image using min/max curvature flow.


    This function directly calls the execute method of BinaryMinMaxCurvatureFlowImageFilter in order to support a procedural API


    See:
     itk::simple::BinaryMinMaxCurvatureFlowImageFilter for the object oriented interface



    
### sitk.BinaryMinMaxCurvatureFlowImageFilter


    Denoise a binary image using min/max curvature flow.


    BinaryMinMaxCurvatureFlowImageFilter implements a curvature driven image denosing algorithm. This filter
    assumes that the image is essentially binary: consisting of two
    classes. Iso-brightness contours in the input image are viewed as a
    level set. The level set is then evolved using a curvature-based speed
    function:

    \[ I_t = F_{\mbox{minmax}} |\nabla I| \]

    where $ F_{\mbox{minmax}} = \min(\kappa,0) $ if $ \mbox{Avg}_{\mbox{stencil}}(x) $ is less than or equal to $ T_{thresold} $ and $ \max(\kappa,0) $ , otherwise. $ \kappa $ is the mean curvature of the iso-brightness contour at point $ x $ .

    In min/max curvature flow, movement is turned on or off depending on
    the scale of the noise one wants to remove. Switching depends on the
    average image value of a region of radius $ R $ around each point. The choice of $ R $ , the stencil radius, governs the scale of the noise to be removed.

    The threshold value $ T_{threshold} $ is a user specified value which discriminates between the two pixel
    classes.

    This filter make use of the multi-threaded finite difference solver
    hierarchy. Updates are computed using a BinaryMinMaxCurvatureFlowFunction object. A zero flux Neumann boundary condition is used when computing
    derivatives near the data boundary.


    WARNING:
    This filter assumes that the input and output types have the same
    dimensions. This filter also requires that the output image pixels are
    of a real type. This filter works for any dimensional images.
     Reference: "Level Set Methods and Fast Marching Methods", J.A.
    Sethian, Cambridge Press, Chapter 16, Second edition, 1999.


    See:
     BinaryMinMaxCurvatureFlowFunction

     CurvatureFlowImageFilter

     MinMaxCurvatureFlowImageFilter

     itk::simple::BinaryMinMaxCurvatureFlow for the procedural interface

     itk::BinaryMinMaxCurvatureFlowImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkBinaryMinMaxCurvatureFlowImageFilter.h

    
### sitk.BinaryMorphologicalClosing
    BinaryMorphologicalClosing(Image arg1, uint32_t radius=1, itk::simple::KernelEnum kernel, double foregroundValue=1.0, bool safeBorder=True) -> Image
    BinaryMorphologicalClosing(Image arg1, VectorUInt32 vectorRadius, itk::simple::KernelEnum kernel, double foregroundValue=1.0, bool safeBorder=True) -> Image



    itk::simple::BinaryMorphologicalClosingImageFilter Functional Interface

    This function directly calls the execute method of BinaryMorphologicalClosingImageFilter in order to support a fully functional API


    
### sitk.BinaryMorphologicalClosingImageFilter


    binary morphological closing of an image.


    This filter removes small (i.e., smaller than the structuring element)
    holes and tube like structures in the interior or at the boundaries of
    the image. The morphological closing of an image "f" is defined as:
    Closing(f) = Erosion(Dilation(f)).

    The structuring element is assumed to be composed of binary values
    (zero or one). Only elements of the structuring element having values
    > 0 are candidates for affecting the center pixel.

    This code was contributed in the Insight Journal paper: "Binary
    morphological closing and opening image filters" by Lehmann G. https://hdl.handle.net/1926/141 http://www.insight-journal.org/browse/publication/58


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.

    See:
     MorphologyImageFilter , GrayscaleDilateImageFilter , GrayscaleErodeImageFilter

     itk::simple::BinaryMorphologicalClosing for the procedural interface

     itk::BinaryMorphologicalClosingImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkBinaryMorphologicalClosingImageFilter.h

    
### sitk.BinaryMorphologicalOpening
    BinaryMorphologicalOpening(Image arg1, uint32_t radius=1, itk::simple::KernelEnum kernel, double backgroundValue=0.0, double foregroundValue=1.0) -> Image
    BinaryMorphologicalOpening(Image arg1, VectorUInt32 vectorRadius, itk::simple::KernelEnum kernel, double backgroundValue=0.0, double foregroundValue=1.0) -> Image



    itk::simple::BinaryMorphologicalOpeningImageFilter Functional Interface

    This function directly calls the execute method of BinaryMorphologicalOpeningImageFilter in order to support a fully functional API


    
### sitk.BinaryMorphologicalOpeningImageFilter


    binary morphological opening of an image.


    This filter removes small (i.e., smaller than the structuring element)
    structures in the interior or at the boundaries of the image. The
    morphological opening of an image "f" is defined as: Opening(f) =
    Dilatation(Erosion(f)).

    The structuring element is assumed to be composed of binary values
    (zero or one). Only elements of the structuring element having values
    > 0 are candidates for affecting the center pixel.

    This code was contributed in the Insight Journal paper: "Binary
    morphological closing and opening image filters" by Lehmann G. https://hdl.handle.net/1926/141 http://www.insight-journal.org/browse/publication/58


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.

    See:
     MorphologyImageFilter , GrayscaleDilateImageFilter , GrayscaleErodeImageFilter

     itk::simple::BinaryMorphologicalOpening for the procedural interface

     itk::BinaryMorphologicalOpeningImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkBinaryMorphologicalOpeningImageFilter.h

    
### sitk.BinaryNot
    BinaryNot(Image image1, double foregroundValue=1.0, double backgroundValue=0.0) -> Image



    Implements the BinaryNot logical operator pixel-wise between two
    images.


    This function directly calls the execute method of BinaryNotImageFilter in order to support a procedural API


    See:
     itk::simple::BinaryNotImageFilter for the object oriented interface



    
### sitk.BinaryNotImageFilter


    Implements the BinaryNot logical operator pixel-wise between two
    images.


    This class is parametrized over the types of the two input images and
    the type of the output image. Numeric conversions (castings) are done
    by the C++ defaults.

    The total operation over one pixel will be

    output_pixel = static_cast<PixelType>( input1_pixel != input2_pixel )

    Where "!=" is the equality operator in C++.


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.
     This implementation was taken from the Insight Journal paper: https://hdl.handle.net/1926/584 or http://www.insight-journal.org/browse/publication/176
    See:
     itk::simple::BinaryNot for the procedural interface

     itk::BinaryNotImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkBinaryNotImageFilter.h

    
### sitk.BinaryOpeningByReconstruction
    BinaryOpeningByReconstruction(Image arg1, uint32_t radius=1, itk::simple::KernelEnum kernel, double foregroundValue=1.0, double backgroundValue=0.0, bool fullyConnected=False) -> Image
    BinaryOpeningByReconstruction(Image arg1, VectorUInt32 vectorRadius, itk::simple::KernelEnum kernel, double foregroundValue=1.0, double backgroundValue=0.0, bool fullyConnected=False) -> Image



    itk::simple::BinaryOpeningByReconstructionImageFilter Functional Interface

    This function directly calls the execute method of BinaryOpeningByReconstructionImageFilter in order to support a fully functional API


    
### sitk.BinaryOpeningByReconstructionImageFilter


    binary morphological closing of an image.


    This filter removes small (i.e., smaller than the structuring element)
    objects in the image. It is defined as: Opening(f) =
    ReconstructionByDilatation(Erosion(f)).

    The structuring element is assumed to be composed of binary values
    (zero or one). Only elements of the structuring element having values
    > 0 are candidates for affecting the center pixel.


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.
     This implementation was taken from the Insight Journal paper: https://hdl.handle.net/1926/584 or http://www.insight-journal.org/browse/publication/176


    See:
     MorphologyImageFilter , OpeningByReconstructionImageFilter , BinaryClosingByReconstructionImageFilter

     itk::simple::BinaryOpeningByReconstruction for the procedural interface

     itk::BinaryOpeningByReconstructionImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkBinaryOpeningByReconstructionImageFilter.h

    
### sitk.BinaryProjection
    BinaryProjection(Image image1, unsigned int projectionDimension=0, double foregroundValue=1.0, double backgroundValue=0.0) -> Image



    Binary projection.


    This function directly calls the execute method of BinaryProjectionImageFilter in order to support a procedural API


    See:
     itk::simple::BinaryProjectionImageFilter for the object oriented interface



    
### sitk.BinaryProjectionImageFilter


    Binary projection.


    This class was contributed to the Insight Journal by Gaetan Lehmann.
    The original paper can be found at https://hdl.handle.net/1926/164


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.

    See:
     ProjectionImageFilter

     MedianProjectionImageFilter

     MeanProjectionImageFilter

     MeanProjectionImageFilter

     MaximumProjectionImageFilter

     MinimumProjectionImageFilter

     StandardDeviationProjectionImageFilter

     SumProjectionImageFilter

     itk::simple::BinaryProjection for the procedural interface

     itk::BinaryProjectionImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkBinaryProjectionImageFilter.h

    
### sitk.BinaryReconstructionByDilation
    BinaryReconstructionByDilation(Image image1, Image image2, double backgroundValue=0.0, double foregroundValue=1.0, bool fullyConnected=False) -> Image



    binary reconstruction by dilation of an image


    This function directly calls the execute method of BinaryReconstructionByDilationImageFilter in order to support a procedural API


    See:
     itk::simple::BinaryReconstructionByDilationImageFilter for the object oriented interface



    
### sitk.BinaryReconstructionByDilationImageFilter


    binary reconstruction by dilation of an image


    Reconstruction by dilation operates on a "marker" image and a
    "mask" image, and is defined as the dilation of the marker image
    with respect to the mask image iterated until stability.

    Geodesic morphology is described in Chapter 6.2 of Pierre Soille's
    book "Morphological Image Analysis: Principles and Applications",
    Second Edition, Springer, 2003.


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.
     This implementation was taken from the Insight Journal paper: https://hdl.handle.net/1926/584 or http://www.insight-journal.org/browse/publication/176


    See:
     MorphologyImageFilter , ReconstructionByDilationImageFilter , BinaryReconstructionByErosionImageFilter

     itk::simple::BinaryReconstructionByDilation for the procedural interface

     itk::BinaryReconstructionByDilationImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkBinaryReconstructionByDilationImageFilter.h

    
### sitk.BinaryReconstructionByErosion
    BinaryReconstructionByErosion(Image image1, Image image2, double backgroundValue=0.0, double foregroundValue=1.0, bool fullyConnected=False) -> Image



    binary reconstruction by erosion of an image


    This function directly calls the execute method of BinaryReconstructionByErosionImageFilter in order to support a procedural API


    See:
     itk::simple::BinaryReconstructionByErosionImageFilter for the object oriented interface



    
### sitk.BinaryReconstructionByErosionImageFilter


    binary reconstruction by erosion of an image


    Reconstruction by erosion operates on a "marker" image and a
    "mask" image, and is defined as the erosion of the marker image with
    respect to the mask image iterated until stability.

    Geodesic morphology is described in Chapter 6.2 of Pierre Soille's
    book "Morphological Image Analysis: Principles and Applications",
    Second Edition, Springer, 2003.


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.
     This implementation was taken from the Insight Journal paper: https://hdl.handle.net/1926/584 or http://www.insight-journal.org/browse/publication/176


    See:
     MorphologyImageFilter , ReconstructionByErosionImageFilter , BinaryReconstructionByDilationImageFilter

     itk::simple::BinaryReconstructionByErosion for the procedural interface

     itk::BinaryReconstructionByErosionImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkBinaryReconstructionByErosionImageFilter.h

    
### sitk.BinaryThinning
    BinaryThinning(Image image1) -> Image



    This filter computes one-pixel-wide edges of the input image.


    This function directly calls the execute method of BinaryThinningImageFilter in order to support a procedural API


    See:
     itk::simple::BinaryThinningImageFilter for the object oriented interface



    
### sitk.BinaryThinningImageFilter


    This filter computes one-pixel-wide edges of the input image.


    This class is parametrized over the type of the input image and the
    type of the output image.

    The input is assumed to be a binary image. If the foreground pixels of
    the input image do not have a value of 1, they are rescaled to 1
    internally to simplify the computation.

    The filter will produce a skeleton of the object. The output
    background values are 0, and the foreground values are 1.

    This filter is a sequential thinning algorithm and known to be
    computational time dependable on the image size. The algorithm
    corresponds with the 2D implementation described in:

    Rafael C. Gonzales and Richard E. Woods. Digital Image Processing. Addison Wesley, 491-494, (1993).

    To do: Make this filter ND.


    See:
     MorphologyImageFilter

     itk::simple::BinaryThinning for the procedural interface

     itk::BinaryThinningImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkBinaryThinningImageFilter.h

    
### sitk.BinaryThreshold
    BinaryThreshold(Image image1, double lowerThreshold=0.0, double upperThreshold=255.0, uint8_t insideValue=1, uint8_t outsideValue=0) -> Image



    Binarize an input image by thresholding.


    This function directly calls the execute method of BinaryThresholdImageFilter in order to support a procedural API


    See:
     itk::simple::BinaryThresholdImageFilter for the object oriented interface



    
### sitk.BinaryThresholdImageFilter


    Binarize an input image by thresholding.


    This filter produces an output image whose pixels are either one of
    two values ( OutsideValue or InsideValue ), depending on whether the
    corresponding input image pixels lie between the two thresholds (
    LowerThreshold and UpperThreshold ). Values equal to either threshold
    is considered to be between the thresholds.

    More precisely \[ Output(x_i) = \begin{cases} InsideValue & \text{if
    \f$LowerThreshold \leq x_i \leq UpperThreshold\f$}
    \\ OutsideValue & \text{otherwise} \end{cases} \]

    This filter is templated over the input image type and the output
    image type.

    The filter expect both images to have the same number of dimensions.

    The default values for LowerThreshold and UpperThreshold are:
    LowerThreshold = NumericTraits<TInput>::NonpositiveMin() ; UpperThreshold = NumericTraits<TInput>::max() ; Therefore, generally only one of these needs to be set, depending
    on whether the user wants to threshold above or below the desired
    threshold.
    See:
     itk::simple::BinaryThreshold for the procedural interface

     itk::BinaryThresholdImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkBinaryThresholdImageFilter.h

    
### sitk.BinaryThresholdProjection
    BinaryThresholdProjection(Image image1, unsigned int projectionDimension=0, double thresholdValue=0.0, uint8_t foregroundValue=1, uint8_t backgroundValue=0) -> Image



    BinaryThreshold projection.


    This function directly calls the execute method of BinaryThresholdProjectionImageFilter in order to support a procedural API


    See:
     itk::simple::BinaryThresholdProjectionImageFilter for the object oriented interface



    
### sitk.BinaryThresholdProjectionImageFilter


    BinaryThreshold projection.


    This class was contributed to the Insight Journal by Gaetan Lehmann.
    the original paper can be found at https://hdl.handle.net/1926/164


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.

    See:
     ProjectionImageFilter

     MedianProjectionImageFilter

     MeanProjectionImageFilter

     MeanProjectionImageFilter

     MaximumProjectionImageFilter

     MinimumProjectionImageFilter

     StandardDeviationProjectionImageFilter

     SumProjectionImageFilter

     itk::simple::BinaryThresholdProjection for the procedural interface

     itk::BinaryThresholdProjectionImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkBinaryThresholdProjectionImageFilter.h

    
### sitk.BinomialBlur
    BinomialBlur(Image image1, unsigned int repetitions=1) -> Image



    Performs a separable blur on each dimension of an image.


    This function directly calls the execute method of BinomialBlurImageFilter in order to support a procedural API


    See:
     itk::simple::BinomialBlurImageFilter for the object oriented interface



    
### sitk.BinomialBlurImageFilter


    Performs a separable blur on each dimension of an image.


    The binomial blur consists of a nearest neighbor average along each
    image dimension. The net result after n-iterations approaches
    convultion with a gaussian.
    See:
     itk::simple::BinomialBlur for the procedural interface

     itk::BinomialBlurImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkBinomialBlurImageFilter.h

    
### sitk.BitwiseNot
    BitwiseNot(Image image1) -> Image



    Implements pixel-wise generic operation on one image.


    This function directly calls the execute method of BitwiseNotImageFilter in order to support a procedural API


    See:
     itk::simple::BitwiseNotImageFilter for the object oriented interface



    
### sitk.BitwiseNotImageFilter


    Implements pixel-wise generic operation on one image.


    This class is parameterized over the type of the input image and the
    type of the output image. It is also parameterized by the operation to
    be applied, using a Functor style.

    UnaryFunctorImageFilter allows the output dimension of the filter to be larger than the input
    dimension. Thus subclasses of the UnaryFunctorImageFilter (like the CastImageFilter ) can be used to promote a 2D image to a 3D image, etc.


    See:
     BinaryFunctorImageFilter TernaryFunctorImageFilter

     itk::simple::BitwiseNot for the procedural interface

     itk::UnaryFunctorImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkBitwiseNotImageFilter.h

    
### sitk.BlackTopHat
    BlackTopHat(Image arg1, uint32_t radius=1, itk::simple::KernelEnum kernel, bool safeBorder=True) -> Image
    BlackTopHat(Image arg1, VectorUInt32 vectorRadius, itk::simple::KernelEnum kernel, bool safeBorder=True) -> Image



    itk::simple::BlackTopHatImageFilter Functional Interface

    This function directly calls the execute method of BlackTopHatImageFilter in order to support a fully functional API


    
### sitk.BlackTopHatImageFilter


    Black top hat extracts local minima that are smaller than the
    structuring element.


    Black top hat extracts local minima that are smaller than the
    structuring element. It subtracts the background from the input image.
    The output of the filter transforms the black valleys into white
    peaks.

    Top-hats are described in Chapter 4.5 of Pierre Soille's book
    "Morphological Image Analysis: Principles and Applications", Second
    Edition, Springer, 2003.


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.

    See:
     itk::simple::BlackTopHat for the procedural interface

     itk::BlackTopHatImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkBlackTopHatImageFilter.h

    
### sitk.BoundedReciprocal
    BoundedReciprocal(Image image1) -> Image



    Computes 1/(1+x) for each pixel in the image.


    This function directly calls the execute method of BoundedReciprocalImageFilter in order to support a procedural API


    See:
     itk::simple::BoundedReciprocalImageFilter for the object oriented interface



    
### sitk.BoundedReciprocalImageFilter


    Computes 1/(1+x) for each pixel in the image.


    The filter expect both the input and output images to have the same
    number of dimensions, and both of a scalar image type.
    See:
     itk::simple::BoundedReciprocal for the procedural interface

     itk::BoundedReciprocalImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkBoundedReciprocalImageFilter.h

    
### sitk.BoxMean
    BoxMean(Image image1, VectorUInt32 radius) -> Image



    Implements a fast rectangular mean filter using the accumulator
    approach.


    This function directly calls the execute method of BoxMeanImageFilter in order to support a procedural API


    See:
     itk::simple::BoxMeanImageFilter for the object oriented interface



    
### sitk.BoxMeanImageFilter


    Implements a fast rectangular mean filter using the accumulator
    approach.


    This code was contributed in the Insight Journal paper: "Efficient
    implementation of kernel filtering" by Beare R., Lehmann G https://hdl.handle.net/1926/555 http://www.insight-journal.org/browse/publication/160


    Richard Beare

    See:
     itk::simple::BoxMean for the procedural interface

     itk::BoxMeanImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkBoxMeanImageFilter.h

    
### sitk.BoxSigma
    BoxSigma(Image image1, VectorUInt32 radius) -> Image



    Implements a fast rectangular sigma filter using the accumulator
    approach.


    This function directly calls the execute method of BoxSigmaImageFilter in order to support a procedural API


    See:
     itk::simple::BoxSigmaImageFilter for the object oriented interface



    
### sitk.BoxSigmaImageFilter


    Implements a fast rectangular sigma filter using the accumulator
    approach.


    This code was contributed in the Insight Journal paper: "Efficient
    implementation of kernel filtering" by Beare R., Lehmann G https://hdl.handle.net/1926/555 http://www.insight-journal.org/browse/publication/160


    Gaetan Lehmann

    See:
     itk::simple::BoxSigma for the procedural interface

     itk::BoxSigmaImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkBoxSigmaImageFilter.h

    
### sitk.CannyEdgeDetection
    CannyEdgeDetection(Image image1, double lowerThreshold=0.0, double upperThreshold=0.0, VectorDouble variance, VectorDouble maximumError) -> Image



    This filter is an implementation of a Canny edge detector for scalar-
    valued images.


    This function directly calls the execute method of CannyEdgeDetectionImageFilter in order to support a procedural API


    See:
     itk::simple::CannyEdgeDetectionImageFilter for the object oriented interface



    
### sitk.CannyEdgeDetectionImageFilter


    This filter is an implementation of a Canny edge detector for scalar-
    valued images.


    Based on John Canny's paper "A Computational Approach to Edge
    Detection"(IEEE Transactions on Pattern Analysis and Machine
    Intelligence, Vol. PAMI-8, No.6, November 1986), there are four major
    steps used in the edge-detection scheme: (1) Smooth the input image
    with Gaussian filter. (2) Calculate the second directional derivatives
    of the smoothed image. (3) Non-Maximum Suppression: the zero-crossings
    of 2nd derivative are found, and the sign of third derivative is used
    to find the correct extrema. (4) The hysteresis thresholding is
    applied to the gradient magnitude (multiplied with zero-crossings) of
    the smoothed image to find and link edges.

    Inputs and Outputs
    The input to this filter should be a scalar, real-valued Itk image of
    arbitrary dimension. The output should also be a scalar, real-value
    Itk image of the same dimensionality.
    Parameters
    There are four parameters for this filter that control the sub-filters
    used by the algorithm.

    Variance and Maximum error are used in the Gaussian smoothing of the
    input image. See itkDiscreteGaussianImageFilter for information on
    these parameters.

    Threshold is the lowest allowed value in the output image. Its data
    type is the same as the data type of the output image. Any values
    below the Threshold level will be replaced with the OutsideValue
    parameter value, whose default is zero.
     TodoEdge-linking will be added when an itk connected component
    labeling algorithm is available.


    See:
     DiscreteGaussianImageFilter

     ZeroCrossingImageFilter

     ThresholdImageFilter

     itk::simple::CannyEdgeDetection for the procedural interface

     itk::CannyEdgeDetectionImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkCannyEdgeDetectionImageFilter.h

    
### sitk.Cast
    Cast(Image image, itk::simple::PixelIDValueEnum pixelID) -> Image



    
### sitk.CastImageFilter


    A hybrid cast image filter to convert images to other types of images.


    Several different ITK classes are implemented under the hood, to
    convert between different image types.


    See:
     itk::simple::Cast for the procedural interface


    C++ includes: sitkCastImageFilter.h

    
### sitk.CenteredTransformInitializer
    CenteredTransformInitializer(Image fixedImage, Image movingImage, Transform transform, itk::simple::CenteredTransformInitializerFilter::OperationModeType operationMode) -> Transform



    CenteredTransformInitializer is a helper class intended to initialize the center of rotation and
    the translation of Transforms having the center of rotation among
    their parameters.


    This function directly calls the execute method of CenteredTransformInitializerFilter in order to support a procedural API


    See:
     itk::simple::CenteredTransformInitializerFilter for the object oriented interface



    
### sitk.CenteredTransformInitializerFilter


    CenteredTransformInitializerFilter is a helper class intended to initialize the center of rotation and
    the translation of Transforms having the center of rotation among
    their parameters.


    This class is connected to the fixed image, moving image and transform
    involved in the registration. Two modes of operation are possible:


    Geometrical,

    Center of mass
     In the first mode, the geometrical center of the moving image is
    passed as initial center of rotation to the transform and the vector
    from the center of the fixed image to the center of the moving image
    is passed as the initial translation. This mode basically assumes that
    the anatomical objects to be registered are centered in their
    respective images. Hence the best initial guess for the registration
    is the one that superimposes those two centers.

    In the second mode, the moments of gray level values are computed for
    both images. The center of mass of the moving image is then used as
    center of rotation. The vector between the two centers of mass is
    passes as the initial translation to the transform. This second
    approach assumes that the moments of the anatomical objects are
    similar for both images and hence the best initial guess for
    registration is to superimpose both mass centers. Note that this
    assumption will probably not hold in multi-modality registration.


    See:
     itk::CenteredTransformInitializer


    C++ includes: sitkCenteredTransformInitializerFilter.h

    
### sitk.CenteredVersorTransformInitializer
    CenteredVersorTransformInitializer(Image fixedImage, Image movingImage, Transform transform, bool computeRotation=False) -> Transform



    CenteredVersorTransformInitializer is a helper class intended to initialize the center of rotation,
    versor, and translation of the VersorRigid3DTransform.


    This function directly calls the execute method of
    CenteredVectorTransformInitializerFilter in order to support a
    procedural API.


    See:
     itk::simple::CenteredVersorTransformInitializerFilter for the object oriented interface



    
### sitk.CenteredVersorTransformInitializerFilter


    CenteredVersorTransformInitializerFilter is a helper class intended to initialize the center of rotation,
    versor, and translation of the VersorRigid3DTransform.


    This class derived from the CenteredTransformInitializerand uses it in
    a more constrained context. It always uses the Moments mode, and also
    takes advantage of the second order moments in order to initialize the
    Versorrepresenting rotation.


    See:
     itk::CenteredVersorTransformInitializer for the Doxygen on the original ITK class.


    C++ includes: sitkCenteredVersorTransformInitializerFilter.h

    
### sitk.ChangeLabel
    ChangeLabel(Image image1, DoubleDoubleMap changeMap) -> Image



    Change Sets of Labels.


    This function directly calls the execute method of ChangeLabelImageFilter in order to support a procedural API


    See:
     itk::simple::ChangeLabelImageFilter for the object oriented interface



    
### sitk.ChangeLabelImageFilter


    Change Sets of Labels.


    This filter produces an output image whose pixels are either copied
    from the input if they are not being changed or are rewritten based on
    the change parameters

    This filter is templated over the input image type and the output
    image type.

    The filter expect both images to have the same number of dimensions.


    Tim Kelliher. GE Research, Niskayuna, NY.

    This work was supported by a grant from DARPA, executed by the U.S.
    Army Medical Research and Materiel Command/TATRC Assistance Agreement,
    Contract::W81XWH-05-2-0059.

    See:
     itk::simple::ChangeLabel for the procedural interface

     itk::ChangeLabelImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkChangeLabelImageFilter.h

    
### sitk.ChangeLabelLabelMap
    ChangeLabelLabelMap(Image image1, DoubleDoubleMap changeMap) -> Image



    Replace the label Ids of selected LabelObjects with new label Ids.


    This function directly calls the execute method of ChangeLabelLabelMapFilter in order to support a procedural API


    See:
     itk::simple::ChangeLabelLabelMapFilter for the object oriented interface



    
### sitk.ChangeLabelLabelMapFilter


    Replace the label Ids of selected LabelObjects with new label Ids.


    This filter takes as input a label map and a list of pairs of Label
    Ids, to produce as output a new label map where the label Ids have
    been replaced according to the pairs in the list.

    Labels that are relabeled to the same label Id are automatically
    merged and optimized into a single LabelObject . The background label can also be changed. Any object relabeled to
    the output background will automatically be removed.

    This implementation was taken from the Insight Journal paper: https://hdl.handle.net/1926/584 or http://www.insight-journal.org/browse/publication/176


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.

    See:
     ShapeLabelObject , RelabelComponentImageFilter , ChangeLabelImageFilter

     itk::simple::ChangeLabelLabelMapFilter for the procedural interface

     itk::ChangeLabelLabelMapFilter for the Doxygen on the original ITK class.


    C++ includes: sitkChangeLabelLabelMapFilter.h

    
### sitk.CheckerBoard
    CheckerBoard(Image image1, Image image2, VectorUInt32 checkerPattern) -> Image



    Combines two images in a checkerboard pattern.


    This function directly calls the execute method of CheckerBoardImageFilter in order to support a procedural API


    See:
     itk::simple::CheckerBoardImageFilter for the object oriented interface



    
### sitk.CheckerBoardImageFilter


    Combines two images in a checkerboard pattern.


    CheckerBoardImageFilter takes two input images that must have the same dimension, size,
    origin and spacing and produces an output image of the same size by
    combinining the pixels from the two input images in a checkerboard
    pattern. This filter is commonly used for visually comparing two
    images, in particular for evaluating the results of an image
    registration process.

    This filter is implemented as a multithreaded filter. It provides a
    ThreadedGenerateData() method for its implementation.
    See:
     itk::simple::CheckerBoard for the procedural interface

     itk::CheckerBoardImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkCheckerBoardImageFilter.h

    
### sitk.Clamp
    Clamp(Image image1, itk::simple::PixelIDValueEnum outputPixelType, double lowerBound, double upperBound) -> Image



    Casts input pixels to output pixel type and clamps the output pixel
    values to a specified range.


    This function directly calls the execute method of ClampImageFilter in order to support a procedural API


    See:
     itk::simple::ClampImageFilter for the object oriented interface



    
### sitk.ClampImageFilter


    Casts input pixels to output pixel type and clamps the output pixel
    values to a specified range.


    Default range corresponds to the range supported by the pixel type of
    the output image.

    This filter is templated over the input image type and the output
    image type.


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.

    See:
     UnaryFunctorImageFilter

     CastImageFilter

     itk::simple::Clamp for the procedural interface

     itk::ClampImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkClampImageFilter.h

    
### sitk.ClosingByReconstruction
    ClosingByReconstruction(Image arg1, uint32_t radius=1, itk::simple::KernelEnum kernel, bool fullyConnected=False, bool preserveIntensities=False) -> Image
    ClosingByReconstruction(Image arg1, VectorUInt32 vectorRadius, itk::simple::KernelEnum kernel, bool fullyConnected=False, bool preserveIntensities=False) -> Image



    itk::simple::ClosingByReconstructionImageFilter Functional Interface

    This function directly calls the execute method of ClosingByReconstructionImageFilter in order to support a fully functional API


    
### sitk.ClosingByReconstructionImageFilter


    Closing by reconstruction of an image.


    This filter is similar to the morphological closing, but contrary to
    the mophological closing, the closing by reconstruction preserves the
    shape of the components. The closing by reconstruction of an image
    "f" is defined as:

    ClosingByReconstruction(f) = ErosionByReconstruction(f, Dilation(f)).

    Closing by reconstruction not only preserves structures preserved by
    the dilation, but also levels raises the contrast of the darkest
    regions. If PreserveIntensities is on, a subsequent reconstruction by
    dilation using a marker image that is the original image for all
    unaffected pixels.

    Closing by reconstruction is described in Chapter 6.3.9 of Pierre
    Soille's book "Morphological Image Analysis: Principles and
    Applications", Second Edition, Springer, 2003.


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.

    See:
     GrayscaleMorphologicalClosingImageFilter

     itk::simple::ClosingByReconstruction for the procedural interface

     itk::ClosingByReconstructionImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkClosingByReconstructionImageFilter.h

    
### sitk.CollidingFronts
    CollidingFronts(Image image1, VectorUIntList seedPoints1, VectorUIntList seedPoints2, bool applyConnectivity=True, double negativeEpsilon=-1e-6, bool stopOnTargets=False) -> Image



    Selects a region of space where two independent fronts run towards
    each other.


    This function directly calls the execute method of CollidingFrontsImageFilter in order to support a procedural API


    See:
     itk::simple::CollidingFrontsImageFilter for the object oriented interface



    
### sitk.CollidingFrontsImageFilter


    Selects a region of space where two independent fronts run towards
    each other.


    The filter can be used to quickly segment anatomical structures (e.g.
    for level set initialization).

    The filter uses two instances of FastMarchingUpwindGradientImageFilter to compute the gradients of arrival times of two wavefronts
    propagating from two sets of seeds. The input of the filter is used as
    the speed of the two wavefronts. The output is the dot product between
    the two gradient vector fields.

    The filter works on the following basic idea. In the regions where the
    dot product between the two gradient fields is negative, the two
    fronts propagate in opposite directions. In the regions where the dot
    product is positive, the two fronts propagate in the same direction.
    This can be used to extract the region of space between two sets of
    points.

    If StopOnTargets is On, then each front will stop as soon as all seeds
    of the other front have been reached. This can markedly speed up the
    execution of the filter, since wave propagation does not take place on
    the complete image.

    Optionally, a connectivity criterion can be applied to the resulting
    dot product image. In this case, the only negative region in the
    output image is the one connected to the seeds.


    Luca Antiga Ph.D. Biomedical Technologies Laboratory, Bioengineering
    Department, Mario Negri Institute, Italy.

    See:
     itk::simple::CollidingFronts for the procedural interface

     itk::CollidingFrontsImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkCollidingFrontsImageFilter.h

    
### sitk.Command


    An implementation of the Command design pattern for callback.


    This class provides a callback mechanism for event that occur from the ProcessObject. These commands can be utilized to observe these events.

    The Command can be created on the stack, and will automatically unregistered it's
    self when destroyed.

    For more information see the page Commands and Events for SimpleITK.

    C++ includes: sitkCommand.h

    
### sitk.ComplexToImaginary
    ComplexToImaginary(Image image1) -> Image



    Computes pixel-wise the imaginary part of a complex image.


    This function directly calls the execute method of ComplexToImaginaryImageFilter in order to support a procedural API


    See:
     itk::simple::ComplexToImaginaryImageFilter for the object oriented interface



    
### sitk.ComplexToImaginaryImageFilter


    Computes pixel-wise the imaginary part of a complex image.



    See:
     itk::simple::ComplexToImaginary for the procedural interface

     itk::ComplexToImaginaryImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkComplexToImaginaryImageFilter.h

    
### sitk.ComplexToModulus
    ComplexToModulus(Image image1) -> Image



    Computes pixel-wise the Modulus of a complex image.


    This function directly calls the execute method of ComplexToModulusImageFilter in order to support a procedural API


    See:
     itk::simple::ComplexToModulusImageFilter for the object oriented interface



    
### sitk.ComplexToModulusImageFilter


    Computes pixel-wise the Modulus of a complex image.



    See:
     itk::simple::ComplexToModulus for the procedural interface

     itk::ComplexToModulusImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkComplexToModulusImageFilter.h

    
### sitk.ComplexToPhase
    ComplexToPhase(Image image1) -> Image



    Computes pixel-wise the modulus of a complex image.


    This function directly calls the execute method of ComplexToPhaseImageFilter in order to support a procedural API


    See:
     itk::simple::ComplexToPhaseImageFilter for the object oriented interface



    
### sitk.ComplexToPhaseImageFilter


    Computes pixel-wise the modulus of a complex image.



    See:
     itk::simple::ComplexToPhase for the procedural interface

     itk::ComplexToPhaseImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkComplexToPhaseImageFilter.h

    
### sitk.ComplexToReal
    ComplexToReal(Image image1) -> Image



    Computes pixel-wise the real(x) part of a complex image.


    This function directly calls the execute method of ComplexToRealImageFilter in order to support a procedural API


    See:
     itk::simple::ComplexToRealImageFilter for the object oriented interface



    
### sitk.ComplexToRealImageFilter


    Computes pixel-wise the real(x) part of a complex image.



    See:
     itk::simple::ComplexToReal for the procedural interface

     itk::ComplexToRealImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkComplexToRealImageFilter.h

    
### sitk.Compose
    Compose(VectorOfImage images) -> Image
    Compose(Image image1) -> Image
    Compose(Image image1, Image image2) -> Image
    Compose(Image image1, Image image2, Image image3) -> Image
    Compose(Image image1, Image image2, Image image3, Image image4) -> Image
    Compose(Image image1, Image image2, Image image3, Image image4, Image image5) -> Image
    
### sitk.ComposeImageFilter


    ComposeImageFilter combine several scalar images into a multicomponent image.


    ComposeImageFilter combine several scalar images into an itk::Image of vector pixel ( itk::Vector , itk::RGBPixel , ...), of std::complex pixel, or in an itk::VectorImage .

    Inputs and Usage
     All input images are expected to have the same template parameters
    and have the same size and origin.

    See:
     VectorImage

     VectorIndexSelectionCastImageFilter

     itk::simple::Compose for the procedural interface


    C++ includes: sitkComposeImageFilter.h

    
### sitk.ConfidenceConnected
    ConfidenceConnected(Image image1, VectorUIntList seedList, unsigned int numberOfIterations=4, double multiplier=4.5, unsigned int initialNeighborhoodRadius=1, uint8_t replaceValue=1) -> Image



    itk::simple::ConfidenceConnectedImageFilter Functional Interface

    This function directly calls the execute method of ConfidenceConnectedImageFilter in order to support a fully functional API


    
### sitk.ConfidenceConnectedImageFilter


    Segment pixels with similar statistics using connectivity.


    This filter extracts a connected set of pixels whose pixel intensities
    are consistent with the pixel statistics of a seed point. The mean and
    variance across a neighborhood (8-connected, 26-connected, etc.) are
    calculated for a seed point. Then pixels connected to this seed point
    whose values are within the confidence interval for the seed point are
    grouped. The width of the confidence interval is controlled by the
    "Multiplier" variable (the confidence interval is the mean plus or
    minus the "Multiplier" times the standard deviation). If the
    intensity variations across a segment were gaussian, a "Multiplier"
    setting of 2.5 would define a confidence interval wide enough to
    capture 99% of samples in the segment.

    After this initial segmentation is calculated, the mean and variance
    are re-calculated. All the pixels in the previous segmentation are
    used to calculate the mean the standard deviation (as opposed to using
    the pixels in the neighborhood of the seed point). The segmentation is
    then recalculated using these refined estimates for the mean and
    variance of the pixel values. This process is repeated for the
    specified number of iterations. Setting the "NumberOfIterations" to
    zero stops the algorithm after the initial segmentation from the seed
    point.

    NOTE: the lower and upper threshold are restricted to lie within the
    valid numeric limits of the input data pixel type. Also, the limits
    may be adjusted to contain the seed point's intensity.
    See:
     itk::simple::ConfidenceConnected for the procedural interface

     itk::ConfidenceConnectedImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkConfidenceConnectedImageFilter.h

    
### sitk.ConnectedComponent
    ConnectedComponent(Image image, Image maskImage, bool fullyConnected=False) -> Image
    ConnectedComponent(Image image, bool fullyConnected=False) -> Image



    
### sitk.ConnectedComponentImageFilter


    Label the objects in a binary image.


    ConnectedComponentImageFilter labels the objects in a binary image (non-zero pixels are considered
    to be objects, zero-valued pixels are considered to be background).
    Each distinct object is assigned a unique label. The filter
    experiments with some improvements to the existing implementation, and
    is based on run length encoding along raster lines. The final object
    labels start with 1 and are consecutive. Objects that are reached
    earlier by a raster order scan have a lower label. This is different
    to the behaviour of the original connected component image filter
    which did not produce consecutive labels or impose any particular
    ordering.

    After the filter is executed, ObjectCount holds the number of
    connected components.


    See:
     ImageToImageFilter

     itk::simple::ConnectedComponent for the procedural interface

     itk::ConnectedComponentImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkConnectedComponentImageFilter.h

    
### sitk.ConnectedThreshold
    ConnectedThreshold(Image image1, VectorUIntList seedList, double lower=0, double upper=1, uint8_t replaceValue=1, itk::simple::ConnectedThresholdImageFilter::ConnectivityType connectivity) -> Image



    itk::simple::ConnectedThresholdImageFilter Functional Interface

    This function directly calls the execute method of ConnectedThresholdImageFilter in order to support a fully functional API


    
### sitk.ConnectedThresholdImageFilter


    Label pixels that are connected to a seed and lie within a range of
    values.


    ConnectedThresholdImageFilter labels pixels with ReplaceValue that are connected to an initial Seed
    AND lie within a Lower and Upper threshold range.
    See:
     itk::simple::ConnectedThreshold for the procedural interface

     itk::ConnectedThresholdImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkConnectedThresholdImageFilter.h

    
### sitk.ConstantPad
    ConstantPad(Image image1, VectorUInt32 padLowerBound, VectorUInt32 padUpperBound, double constant=0.0) -> Image



    Increase the image size by padding with a constant value.


    This function directly calls the execute method of ConstantPadImageFilter in order to support a procedural API


    See:
     itk::simple::ConstantPadImageFilter for the object oriented interface



    
### sitk.ConstantPadImageFilter


    Increase the image size by padding with a constant value.


    ConstantPadImageFilter changes the output image region. If the output image region is larger
    than the input image region, the extra pixels are filled in by a
    constant value. The output image region must be specified.

    Visual explanation of padding regions. This filter is implemented as a
    multithreaded filter. It provides a ThreadedGenerateData() method for
    its implementation.


    See:
     WrapPadImageFilter , MirrorPadImageFilter

     itk::simple::ConstantPad for the procedural interface

     itk::ConstantPadImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkConstantPadImageFilter.h

    
### sitk.Convolution
    Convolution(Image image, Image kernelImage, bool normalize=False, itk::simple::ConvolutionImageFilter::BoundaryConditionType boundaryCondition, itk::simple::ConvolutionImageFilter::OutputRegionModeType outputRegionMode) -> Image



    Convolve a given image with an arbitrary image kernel.


    This function directly calls the execute method of ConvolutionImageFilter in order to support a procedural API


    See:
     itk::simple::ConvolutionImageFilter for the object oriented interface



    
### sitk.ConvolutionImageFilter


    Convolve a given image with an arbitrary image kernel.


    This filter operates by centering the flipped kernel at each pixel in
    the image and computing the inner product between pixel values in the
    image and pixel values in the kernel. The center of the kernel is
    defined as $ \lfloor (2*i+s-1)/2 \rfloor $ where $i$ is the index and $s$ is the size of the largest possible region of the kernel image. For
    kernels with odd sizes in all dimensions, this corresponds to the
    center pixel. If a dimension of the kernel image has an even size,
    then the center index of the kernel in that dimension will be the
    largest integral index that is less than the continuous index of the
    image center.

    The kernel can optionally be normalized to sum to 1 using NormalizeOn() . Normalization is off by default.


    WARNING:
    This filter ignores the spacing, origin, and orientation of the kernel
    image and treats them as identical to those in the input image.
     This code was contributed in the Insight Journal paper:

    "Image Kernel Convolution" by Tustison N., Gee J. https://hdl.handle.net/1926/1323 http://www.insight-journal.org/browse/publication/208


    Nicholas J. Tustison

    James C. Gee

    See:
     itk::simple::Convolution for the procedural interface

     itk::ConvolutionImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkConvolutionImageFilter.h

    
### sitk.Cos
    Cos(Image image1) -> Image



    Computes the cosine of each pixel.


    This function directly calls the execute method of CosImageFilter in order to support a procedural API


    See:
     itk::simple::CosImageFilter for the object oriented interface



    
### sitk.CosImageFilter


    Computes the cosine of each pixel.


    This filter is templated over the pixel type of the input image and
    the pixel type of the output image.

    The filter walks over all of the pixels in the input image, and for
    each pixel does the following:


    cast the pixel value to double ,

    apply the std::cos() function to the double value,

    cast the double value resulting from std::cos() to the pixel type of
    the output image,

    store the cast value into the output image.
     The filter expects both images to have the same dimension (e.g. both
    2D, or both 3D, or both ND)
    See:
     itk::simple::Cos for the procedural interface

     itk::CosImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkCosImageFilter.h

    
### sitk.Crop
    Crop(Image image1, VectorUInt32 lowerBoundaryCropSize, VectorUInt32 upperBoundaryCropSize) -> Image



    Decrease the image size by cropping the image by an itk::Size at both the upper and lower bounds of the largest possible region.


    This function directly calls the execute method of CropImageFilter in order to support a procedural API


    See:
     itk::simple::CropImageFilter for the object oriented interface



    
### sitk.CropImageFilter


    Decrease the image size by cropping the image by an itk::Size at both the upper and lower bounds of the largest possible region.


    CropImageFilter changes the image boundary of an image by removing pixels outside the
    target region. The target region is not specified in advance, but
    calculated in BeforeThreadedGenerateData() .

    This filter uses ExtractImageFilter to perform the cropping.
    See:
     itk::simple::Crop for the procedural interface

     itk::CropImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkCropImageFilter.h

    
### sitk.CurvatureAnisotropicDiffusion
    CurvatureAnisotropicDiffusion(Image image1, double timeStep=0.0625, double conductanceParameter=3, unsigned int conductanceScalingUpdateInterval=1, uint32_t numberOfIterations=5) -> Image



    itk::simple::CurvatureAnisotropicDiffusionImageFilter Procedural Interface


    This function directly calls the execute method of CurvatureAnisotropicDiffusionImageFilter in order to support a procedural API


    See:
     itk::simple::CurvatureAnisotropicDiffusionImageFilter for the object oriented interface



    
### sitk.CurvatureAnisotropicDiffusionImageFilter


    This filter performs anisotropic diffusion on a scalar itk::Image using the modified curvature diffusion equation (MCDE) implemented in
    itkCurvatureNDAnisotropicDiffusionFunction. For detailed information
    on anisotropic diffusion and the MCDE see
    itkAnisotropicDiffusionFunction and
    itkCurvatureNDAnisotropicDiffusionFunction.

    Inputs and Outputs
    The input and output to this filter must be a scalar itk::Image with numerical pixel types (float or double). A user defined type
    which correctly defines arithmetic operations with floating point
    accuracy should also give correct results.
    Parameters
    Please first read all the documentation found in AnisotropicDiffusionImageFilter and AnisotropicDiffusionFunction . Also see CurvatureNDAnisotropicDiffusionFunction .
     The default time step for this filter is set to the maximum
    theoretically stable value: 0.5 / 2^N, where N is the dimensionality
    of the image. For a 2D image, this means valid time steps are below
    0.1250. For a 3D image, valid time steps are below 0.0625.


    See:
     AnisotropicDiffusionImageFilter

     AnisotropicDiffusionFunction

     CurvatureNDAnisotropicDiffusionFunction

     itk::simple::CurvatureAnisotropicDiffusion for the procedural interface

     itk::CurvatureAnisotropicDiffusionImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkCurvatureAnisotropicDiffusionImageFilter.h

    
### sitk.CurvatureFlow
    CurvatureFlow(Image image1, double timeStep=0.05, uint32_t numberOfIterations=5) -> Image



    Denoise an image using curvature driven flow.


    This function directly calls the execute method of CurvatureFlowImageFilter in order to support a procedural API


    See:
     itk::simple::CurvatureFlowImageFilter for the object oriented interface



    
### sitk.CurvatureFlowImageFilter


    Denoise an image using curvature driven flow.


    CurvatureFlowImageFilter implements a curvature driven image denoising algorithm. Iso-
    brightness contours in the grayscale input image are viewed as a level
    set. The level set is then evolved using a curvature-based speed
    function:

    \[ I_t = \kappa |\nabla I| \] where $ \kappa $ is the curvature.

    The advantage of this approach is that sharp boundaries are preserved
    with smoothing occurring only within a region. However, it should be
    noted that continuous application of this scheme will result in the
    eventual removal of all information as each contour shrinks to zero
    and disappear.

    Note that unlike level set segmentation algorithms, the image to be
    denoised is already the level set and can be set directly as the input
    using the SetInput() method.

    This filter has two parameters: the number of update iterations to be
    performed and the timestep between each update.

    The timestep should be "small enough" to ensure numerical stability.
    Stability is guarantee when the timestep meets the CFL (Courant-
    Friedrichs-Levy) condition. Broadly speaking, this condition ensures
    that each contour does not move more than one grid position at each
    timestep. In the literature, the timestep is typically user specified
    and have to manually tuned to the application.

    This filter make use of the multi-threaded finite difference solver
    hierarchy. Updates are computed using a CurvatureFlowFunction object. A zero flux Neumann boundary condition when computing
    derivatives near the data boundary.

    This filter may be streamed. To support streaming this filter produces
    a padded output which takes into account edge effects. The size of the
    padding is m_NumberOfIterations on each edge. Users of this filter
    should only make use of the center valid central region.


    WARNING:
    This filter assumes that the input and output types have the same
    dimensions. This filter also requires that the output image pixels are
    of a floating point type. This filter works for any dimensional
    images.
     Reference: "Level Set Methods and Fast Marching Methods", J.A.
    Sethian, Cambridge Press, Chapter 16, Second edition, 1999.


    See:
     DenseFiniteDifferenceImageFilter

     CurvatureFlowFunction

     MinMaxCurvatureFlowImageFilter

     BinaryMinMaxCurvatureFlowImageFilter

     itk::simple::CurvatureFlow for the procedural interface

     itk::CurvatureFlowImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkCurvatureFlowImageFilter.h

    
### sitk.CyclicShift
    CyclicShift(Image image1, VectorInt32 shift) -> Image



    Perform a cyclic spatial shift of image intensities on the image grid.


    This function directly calls the execute method of CyclicShiftImageFilter in order to support a procedural API


    See:
     itk::simple::CyclicShiftImageFilter for the object oriented interface



    
### sitk.CyclicShiftImageFilter


    Perform a cyclic spatial shift of image intensities on the image grid.


    This filter supports arbitrary cyclic shifts of pixel values on the
    image grid. If the Shift is set to [xOff, yOff], the value of the
    pixel at [0, 0] in the input image will be the value of the pixel in
    the output image at index [xOff modulo xSize, yOff modulo ySize] where
    xSize and ySize are the sizes of the image in the x and y dimensions,
    respectively. If a pixel value is moved across a boundary, the pixel
    value is wrapped around that boundary. For example, if the image is
    40-by-40 and the Shift is [13, 47], then the value of the pixel at [0,
    0] in the input image will be the value of the pixel in the output
    image at index [13, 7].

    Negative Shifts are supported. This filter also works with images
    whose largest possible region starts at a non-zero index.
    See:
     itk::simple::CyclicShift for the procedural interface

     itk::CyclicShiftImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkCyclicShiftImageFilter.h

    
### sitk.DanielssonDistanceMap
    DanielssonDistanceMap(Image image1, bool inputIsBinary=False, bool squaredDistance=False, bool useImageSpacing=False) -> Image



    This filter computes the distance map of the input image as an
    approximation with pixel accuracy to the Euclidean distance.


    This function directly calls the execute method of DanielssonDistanceMapImageFilter in order to support a procedural API


    See:
     itk::simple::DanielssonDistanceMapImageFilter for the object oriented interface



    
### sitk.DanielssonDistanceMapImageFilter


    This filter computes the distance map of the input image as an
    approximation with pixel accuracy to the Euclidean distance.


    TInputImage

    Input Image Type

    TOutputImage

    Output Image Type

    TVoronoiImage

    Voronoi Image Type. Note the default value is TInputImage.

    The input is assumed to contain numeric codes defining objects. The
    filter will produce as output the following images:


    A Voronoi partition using the same numeric codes as the input.

    A distance map with the approximation to the euclidean distance. from
    a particular pixel to the nearest object to this pixel in the input
    image.

    A vector map containing the component of the vector relating the
    current pixel with the closest point of the closest object to this
    pixel. Given that the components of the distance are computed in
    "pixels", the vector is represented by an itk::Offset . That is, physical coordinates are not used.
     This filter is N-dimensional and known to be efficient in
    computational time. The algorithm is the N-dimensional version of the
    4SED algorithm given for two dimensions in:

    Danielsson, Per-Erik. Euclidean Distance Mapping. Computer Graphics
    and Image Processing 14, 227-248 (1980).
    See:
     itk::simple::DanielssonDistanceMap for the procedural interface

     itk::DanielssonDistanceMapImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkDanielssonDistanceMapImageFilter.h

    
### sitk.DemonsRegistrationFilter


    Deformably register two images using the demons algorithm.


    DemonsRegistrationFilter implements the demons deformable algorithm that register two images
    by computing the displacement field which will map a moving image onto
    a fixed image.

    A displacement field is represented as a image whose pixel type is
    some vector type with at least N elements, where N is the dimension of
    the fixed image. The vector type must support element access via
    operator []. It is assumed that the vector elements behave like
    floating point scalars.

    This class is templated over the fixed image type, moving image type
    and the displacement field type.

    The input fixed and moving images are set via methods SetFixedImage
    and SetMovingImage respectively. An initial displacement field maybe
    set via SetInitialDisplacementField or SetInput. If no initial field
    is set, a zero field is used as the initial condition.

    The algorithm has one parameters: the number of iteration to be
    performed.

    The output displacement field can be obtained via methods GetOutput or
    GetDisplacementField.

    This class make use of the finite difference solver hierarchy. Update
    for each iteration is computed in DemonsRegistrationFunction .


    WARNING:
    This filter assumes that the fixed image type, moving image type and
    displacement field type all have the same number of dimensions.

    See:
     DemonsRegistrationFunction

     itk::DemonsRegistrationFilter for the Doxygen on the original ITK class.


    C++ includes: sitkDemonsRegistrationFilter.h

    
### sitk.Derivative
    Derivative(Image image1, unsigned int direction=0, unsigned int order=1, bool useImageSpacing=True) -> Image



    Computes the directional derivative of an image. The directional
    derivative at each pixel location is computed by convolution with a
    derivative operator of user-specified order.


    This function directly calls the execute method of DerivativeImageFilter in order to support a procedural API


    See:
     itk::simple::DerivativeImageFilter for the object oriented interface



    
### sitk.DerivativeImageFilter


    Computes the directional derivative of an image. The directional
    derivative at each pixel location is computed by convolution with a
    derivative operator of user-specified order.


    SetOrder specifies the order of the derivative.

    SetDirection specifies the direction of the derivative with respect to
    the coordinate axes of the image.


    See:
     Image

     Neighborhood

     NeighborhoodOperator

     NeighborhoodIterator

     itk::simple::Derivative for the procedural interface

     itk::DerivativeImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkDerivativeImageFilter.h

    
### sitk.DiffeomorphicDemonsRegistrationFilter


    Deformably register two images using a diffeomorphic demons algorithm.


    This class was contributed by Tom Vercauteren, INRIA & Mauna Kea
    Technologies, based on a variation of the DemonsRegistrationFilter . The basic modification is to use diffeomorphism exponentials.

    See T. Vercauteren, X. Pennec, A. Perchant and N. Ayache, "Non-
    parametric Diffeomorphic Image Registration with the Demons
    Algorithm", Proc. of MICCAI 2007.

    DiffeomorphicDemonsRegistrationFilter implements the demons deformable algorithm that register two images
    by computing the deformation field which will map a moving image onto
    a fixed image.

    A deformation field is represented as a image whose pixel type is some
    vector type with at least N elements, where N is the dimension of the
    fixed image. The vector type must support element access via operator
    []. It is assumed that the vector elements behave like floating point
    scalars.

    This class is templated over the fixed image type, moving image type
    and the deformation field type.

    The input fixed and moving images are set via methods SetFixedImage
    and SetMovingImage respectively. An initial deformation field maybe
    set via SetInitialDisplacementField or SetInput. If no initial field
    is set, a zero field is used as the initial condition.

    The output deformation field can be obtained via methods GetOutput or
    GetDisplacementField.

    This class make use of the finite difference solver hierarchy. Update
    for each iteration is computed in DemonsRegistrationFunction .


    Tom Vercauteren, INRIA & Mauna Kea Technologies

    WARNING:
    This filter assumes that the fixed image type, moving image type and
    deformation field type all have the same number of dimensions.
     This implementation was taken from the Insight Journal paper: https://hdl.handle.net/1926/510


    See:
     DemonsRegistrationFilter

     DemonsRegistrationFunction

     itk::DiffeomorphicDemonsRegistrationFilter for the Doxygen on the original ITK class.


    C++ includes: sitkDiffeomorphicDemonsRegistrationFilter.h

    
### sitk.DilateObjectMorphology
    DilateObjectMorphology(Image arg1, uint32_t radius=1, itk::simple::KernelEnum kernel, double objectValue=1) -> Image
    DilateObjectMorphology(Image arg1, VectorUInt32 vectorRadius, itk::simple::KernelEnum kernel, double objectValue=1) -> Image



    itk::simple::DilateObjectMorphologyImageFilter Functional Interface

    This function directly calls the execute method of DilateObjectMorphologyImageFilter in order to support a fully functional API


    
### sitk.DilateObjectMorphologyImageFilter


    dilation of an object in an image


    Dilate an image using binary morphology. Pixel values matching the
    object value are considered the "foreground" and all other pixels
    are "background". This is useful in processing mask images
    containing only one object.

    If a pixel's value is equal to the object value and the pixel is
    adjacent to a non-object valued pixel, then the kernel is centered on
    the object-value pixel and neighboring pixels covered by the kernel
    are assigned the object value. The structuring element is assumed to
    be composed of binary values (zero or one).


    See:
     ObjectMorphologyImageFilter , ErodeObjectMorphologyImageFilter

     BinaryDilateImageFilter

     itk::simple::DilateObjectMorphology for the procedural interface

     itk::DilateObjectMorphologyImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkDilateObjectMorphologyImageFilter.h

    
### sitk.DiscreteGaussian
    DiscreteGaussian(Image image1, double variance, unsigned int maximumKernelWidth=32, double maximumError=0.01, bool useImageSpacing=True) -> Image
    DiscreteGaussian(Image image1, VectorDouble variance, unsigned int maximumKernelWidth=32, VectorDouble maximumError, bool useImageSpacing=True) -> Image



    Blurs an image by separable convolution with discrete gaussian
    kernels. This filter performs Gaussian blurring by separable
    convolution of an image and a discrete Gaussian operator (kernel).


    This function directly calls the execute method of DiscreteGaussianImageFilter in order to support a procedural API


    See:
     itk::simple::DiscreteGaussianImageFilter for the object oriented interface



    
### sitk.DiscreteGaussianDerivative
    DiscreteGaussianDerivative(Image image1, VectorDouble variance, VectorUInt32 order, unsigned int maximumKernelWidth=32, double maximumError=0.01, bool useImageSpacing=True, bool normalizeAcrossScale=False) -> Image



    Calculates image derivatives using discrete derivative gaussian
    kernels. This filter calculates Gaussian derivative by separable
    convolution of an image and a discrete Gaussian derivative operator
    (kernel).


    This function directly calls the execute method of DiscreteGaussianDerivativeImageFilter in order to support a procedural API


    See:
     itk::simple::DiscreteGaussianDerivativeImageFilter for the object oriented interface



    
### sitk.DiscreteGaussianDerivativeImageFilter


    Calculates image derivatives using discrete derivative gaussian
    kernels. This filter calculates Gaussian derivative by separable
    convolution of an image and a discrete Gaussian derivative operator
    (kernel).


    The Gaussian operators used here were described by Tony Lindeberg
    (Discrete Scale-Space Theory and the Scale-Space Primal Sketch.
    Dissertation. Royal Institute of Technology, Stockholm, Sweden. May
    1991.)

    The variance or standard deviation (sigma) will be evaluated as pixel
    units if SetUseImageSpacing is off (false) or as physical units if
    SetUseImageSpacing is on (true, default). The variance can be set
    independently in each dimension.

    When the Gaussian kernel is small, this filter tends to run faster
    than itk::RecursiveGaussianImageFilter .


    Ivan Macia, VICOMTech, Spain, http://www.vicomtech.es
     This implementation was taken from the Insight Journal paper: https://hdl.handle.net/1926/1290


    See:
     GaussianDerivativeOperator

     Image

     Neighborhood

     NeighborhoodOperator

     itk::simple::DiscreteGaussianDerivative for the procedural interface

     itk::DiscreteGaussianDerivativeImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkDiscreteGaussianDerivativeImageFilter.h

    
### sitk.DiscreteGaussianImageFilter


    Blurs an image by separable convolution with discrete gaussian
    kernels. This filter performs Gaussian blurring by separable
    convolution of an image and a discrete Gaussian operator (kernel).


    The Gaussian operator used here was described by Tony Lindeberg
    (Discrete Scale-Space Theory and the Scale-Space Primal Sketch.
    Dissertation. Royal Institute of Technology, Stockholm, Sweden. May
    1991.) The Gaussian kernel used here was designed so that smoothing
    and derivative operations commute after discretization.

    The variance or standard deviation (sigma) will be evaluated as pixel
    units if SetUseImageSpacing is off (false) or as physical units if
    SetUseImageSpacing is on (true, default). The variance can be set
    independently in each dimension.

    When the Gaussian kernel is small, this filter tends to run faster
    than itk::RecursiveGaussianImageFilter .


    See:
     GaussianOperator

     Image

     Neighborhood

     NeighborhoodOperator

     RecursiveGaussianImageFilter

     itk::simple::DiscreteGaussian for the procedural interface

     itk::DiscreteGaussianImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkDiscreteGaussianImageFilter.h

    
### sitk.DisplacementFieldJacobianDeterminant
    DisplacementFieldJacobianDeterminant(Image image1, bool useImageSpacing=True, VectorDouble derivativeWeights) -> Image



    Computes a scalar image from a vector image (e.g., deformation field)
    input, where each output scalar at each pixel is the Jacobian
    determinant of the vector field at that location. This calculation is
    correct in the case where the vector image is a "displacement" from
    the current location. The computation for the jacobian determinant is:
    det[ dT/dx ] = det[ I + du/dx ].


    This function directly calls the execute method of DisplacementFieldJacobianDeterminantFilter in order to support a procedural API


    See:
     itk::simple::DisplacementFieldJacobianDeterminantFilter for the object oriented interface



    
### sitk.DisplacementFieldJacobianDeterminantFilter


    Computes a scalar image from a vector image (e.g., deformation field)
    input, where each output scalar at each pixel is the Jacobian
    determinant of the vector field at that location. This calculation is
    correct in the case where the vector image is a "displacement" from
    the current location. The computation for the jacobian determinant is:
    det[ dT/dx ] = det[ I + du/dx ].


    Overview
    This filter is based on itkVectorGradientMagnitudeImageFilter and
    supports the m_DerivativeWeights weights for partial derivatives.
     Note that the determinant of a zero vector field is also zero,
    whereas the Jacobian determinant of the corresponding identity warp
    transformation is 1.0. In order to compute the effective deformation
    Jacobian determinant 1.0 must be added to the diagonal elements of
    Jacobian prior to taking the derivative. i.e. det([ (1.0+dx/dx) dx/dy
    dx/dz ; dy/dx (1.0+dy/dy) dy/dz; dz/dx dz/dy (1.0+dz/dz) ])

    Template Parameters (Input and Output)
    This filter has one required template parameter which defines the
    input image type. The pixel type of the input image is assumed to be a
    vector (e.g., itk::Vector , itk::RGBPixel , itk::FixedArray ). The scalar type of the vector components must be castable to
    floating point. Instantiating with an image of RGBPixel<unsigned
    short>, for example, is allowed, but the filter will convert it to an
    image of Vector<float,3> for processing.
     The second template parameter, TRealType, can be optionally specified
    to define the scalar numerical type used in calculations. This is the
    component type of the output image, which will be of
    itk::Vector<TRealType, N>, where N is the number of channels in the
    multiple component input image. The default type of TRealType is
    float. For extra precision, you may safely change this parameter to
    double.

    The third template parameter is the output image type. The third
    parameter will be automatically constructed from the first and second
    parameters, so it is not necessary (or advisable) to set this
    parameter explicitly. Given an M-channel input image with
    dimensionality N, and a numerical type specified as TRealType, the
    output image will be of type itk::Image<TRealType, N>.

    Filter Parameters
    The method SetUseImageSpacingOn will cause derivatives in the image to
    be scaled (inversely) with the pixel size of the input image,
    effectively taking derivatives in world coordinates (versus isotropic
    image space). SetUseImageSpacingOff turns this functionality off.
    Default is UseImageSpacingOn. The parameter UseImageSpacing can be set
    directly with the method SetUseImageSpacing(bool) .
     Weights can be applied to the derivatives directly using the
    SetDerivativeWeights method. Note that if UseImageSpacing is set to
    TRUE (ON), then these weights will be overridden by weights derived
    from the image spacing when the filter is updated. The argument to
    this method is a C array of TRealValue type.

    Constraints
    We use vnl_det for determinent computation, which only supports square
    matrices. So the vector dimension of the input image values must be
    equal to the image dimensions, which is trivially true for a
    deformation field that maps an n-dimensional space onto itself.
     Currently, dimensions up to and including 4 are supported. This
    limitation comes from the presence of vnl_det() functions for matrices
    of dimension up to 4x4.

    The template parameter TRealType must be floating point (float or
    double) or a user-defined "real" numerical type with arithmetic
    operations defined sufficient to compute derivatives.


    See:
     Image

     Neighborhood

     NeighborhoodOperator

     NeighborhoodIterator

    This class was adapted by

    Hans J. Johnson, The University of Iowa from code provided by

    Tom Vercauteren, INRIA & Mauna Kea Technologies

    Torsten Rohlfing, Neuroscience Program, SRI International.

    See:
     itk::simple::DisplacementFieldJacobianDeterminantFilter for the procedural interface

     itk::DisplacementFieldJacobianDeterminantFilter for the Doxygen on the original ITK class.


    C++ includes: sitkDisplacementFieldJacobianDeterminantFilter.h

    
### sitk.DisplacementFieldTransform


    A dense deformable transform over a bounded spatial domain for 2D or
    3D coordinates space.



    See:
     itk::DisplacementFieldTransform


    C++ includes: sitkDisplacementFieldTransform.h

    
### sitk.Divide
    Divide(Image image1, Image image2) -> Image
    Divide(Image image1, double constant) -> Image
    Divide(double constant, Image image2) -> Image



    
### sitk.DivideFloor
    DivideFloor(Image image1, Image image2) -> Image
    DivideFloor(Image image1, double constant) -> Image
    DivideFloor(double constant, Image image2) -> Image



    
### sitk.DivideFloorImageFilter


    Implements pixel-wise generic operation of two images, or of an image
    and a constant.


    This class is parameterized over the types of the two input images and
    the type of the output image. It is also parameterized by the
    operation to be applied. A Functor style is used.

    The constant must be of the same type than the pixel type of the
    corresponding image. It is wrapped in a SimpleDataObjectDecorator so it can be updated through the pipeline. The SetConstant() and
    GetConstant() methods are provided as shortcuts to set or get the
    constant value without manipulating the decorator.


    See:
     UnaryFunctorImageFilter TernaryFunctorImageFilter

     itk::simple::DivideFloor for the procedural interface

     itk::BinaryFunctorImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkDivideFloorImageFilter.h

    
### sitk.DivideImageFilter


    Pixel-wise division of two images.


    This class is templated over the types of the two input images and the
    type of the output image. When the divisor is zero, the division
    result is set to the maximum number that can be represented by default
    to avoid exception. Numeric conversions (castings) are done by the C++
    defaults.
    See:
     itk::simple::Divide for the procedural interface

     itk::DivideImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkDivideImageFilter.h

    
### sitk.DivideReal
    DivideReal(Image image1, Image image2) -> Image
    DivideReal(Image image1, double constant) -> Image
    DivideReal(double constant, Image image2) -> Image



    
### sitk.DivideRealImageFilter


    Implements pixel-wise generic operation of two images, or of an image
    and a constant.


    This class is parameterized over the types of the two input images and
    the type of the output image. It is also parameterized by the
    operation to be applied. A Functor style is used.

    The constant must be of the same type than the pixel type of the
    corresponding image. It is wrapped in a SimpleDataObjectDecorator so it can be updated through the pipeline. The SetConstant() and
    GetConstant() methods are provided as shortcuts to set or get the
    constant value without manipulating the decorator.


    See:
     UnaryFunctorImageFilter TernaryFunctorImageFilter

     itk::simple::DivideReal for the procedural interface

     itk::BinaryFunctorImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkDivideRealImageFilter.h

    
### sitk.DoubleDoubleMapProxy of C++ std::map<(double,double)> class.
### sitk.DoubleThreshold
    DoubleThreshold(Image image1, double threshold1=0.0, double threshold2=1.0, double threshold3=254.0, double threshold4=255.0, uint8_t insideValue=1, uint8_t outsideValue=0, bool fullyConnected=False) -> Image



    Binarize an input image using double thresholding.


    This function directly calls the execute method of DoubleThresholdImageFilter in order to support a procedural API


    See:
     itk::simple::DoubleThresholdImageFilter for the object oriented interface



    
### sitk.DoubleThresholdImageFilter


    Binarize an input image using double thresholding.


    Double threshold addresses the difficulty in selecting a threshold
    that will select the objects of interest without selecting extraneous
    objects. Double threshold considers two threshold ranges: a narrow
    range and a wide range (where the wide range encompasses the narrow
    range). If the wide range was used for a traditional threshold (where
    values inside the range map to the foreground and values outside the
    range map to the background), many extraneous pixels may survive the
    threshold operation. If the narrow range was used for a traditional
    threshold, then too few pixels may survive the threshold.

    Double threshold uses the narrow threshold image as a marker image and
    the wide threshold image as a mask image in the geodesic dilation.
    Essentially, the marker image (narrow threshold) is dilated but
    constrained to lie within the mask image (wide threshold). Thus, only
    the objects of interest (those pixels that survived the narrow
    threshold) are extracted but the those objects appear in the final
    image as they would have if the wide threshold was used.


    See:
     GrayscaleGeodesicDilateImageFilter

     MorphologyImageFilter , GrayscaleDilateImageFilter , GrayscaleFunctionDilateImageFilter , BinaryDilateImageFilter

     itk::simple::DoubleThreshold for the procedural interface

     itk::DoubleThresholdImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkDoubleThresholdImageFilter.h

    
### sitk.EdgePotential
    EdgePotential(Image image1) -> Image



    Computes the edge potential of an image from the image gradient.


    This function directly calls the execute method of EdgePotentialImageFilter in order to support a procedural API


    See:
     itk::simple::EdgePotentialImageFilter for the object oriented interface



    
### sitk.EdgePotentialImageFilter


    Computes the edge potential of an image from the image gradient.


    Input to this filter should be a CovariantVector image representing the image gradient.

    The filter expect both the input and output images to have the same
    number of dimensions, and the output to be of a scalar image type.
    See:
     itk::simple::EdgePotential for the procedural interface

     itk::EdgePotentialImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkEdgePotentialImageFilter.h

    
### sitk.Equal
    Equal(Image image1, Image image2, uint8_t backgroundValue=0, uint8_t foregroundValue=1) -> Image
    Equal(Image image1, double constant, uint8_t backgroundValue=0, uint8_t foregroundValue=1) -> Image
    Equal(double constant, Image image2, uint8_t backgroundValue=0, uint8_t foregroundValue=1) -> Image



    
### sitk.EqualImageFilter


    Implements pixel-wise generic operation of two images, or of an image
    and a constant.


    This class is parameterized over the types of the two input images and
    the type of the output image. It is also parameterized by the
    operation to be applied. A Functor style is used.

    The constant must be of the same type than the pixel type of the
    corresponding image. It is wrapped in a SimpleDataObjectDecorator so it can be updated through the pipeline. The SetConstant() and
    GetConstant() methods are provided as shortcuts to set or get the
    constant value without manipulating the decorator.


    See:
     UnaryFunctorImageFilter TernaryFunctorImageFilter

     itk::simple::Equal for the procedural interface

     itk::BinaryFunctorImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkEqualImageFilter.h

    
### sitk.ErodeObjectMorphology
    ErodeObjectMorphology(Image arg1, uint32_t radius=1, itk::simple::KernelEnum kernel, double objectValue=1, double backgroundValue=0) -> Image
    ErodeObjectMorphology(Image arg1, VectorUInt32 vectorRadius, itk::simple::KernelEnum kernel, double objectValue=1, double backgroundValue=0) -> Image



    itk::simple::ErodeObjectMorphologyImageFilter Functional Interface

    This function directly calls the execute method of ErodeObjectMorphologyImageFilter in order to support a fully functional API


    
### sitk.ErodeObjectMorphologyImageFilter


    Erosion of an object in an image.


    Erosion of an image using binary morphology. Pixel values matching the
    object value are considered the "object" and all other pixels are
    "background". This is useful in processing mask images containing
    only one object.

    If the pixel covered by the center of the kernel has the pixel value
    ObjectValue and the pixel is adjacent to a non-object valued pixel,
    then the kernel is centered on the object-value pixel and neighboring
    pixels covered by the kernel are assigned the background value. The
    structuring element is assumed to be composed of binary values (zero
    or one).


    See:
     ObjectMorphologyImageFilter , BinaryFunctionErodeImageFilter

     BinaryErodeImageFilter

     itk::simple::ErodeObjectMorphology for the procedural interface

     itk::ErodeObjectMorphologyImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkErodeObjectMorphologyImageFilter.h

    
### sitk.Euler2DTransform


    A rigid 2D transform with rotation in radians around a fixed center
    with translation.



    See:
     itk::Euler2DTransform


    C++ includes: sitkEuler2DTransform.h

    
### sitk.Euler3DTransform


    A rigid 3D transform with rotation in radians around a fixed center
    with translation.



    See:
     itk::Euler3DTransform


    C++ includes: sitkEuler3DTransform.h

    
### sitk.Exp
    Exp(Image image1) -> Image



    Computes the exponential function of each pixel.


    This function directly calls the execute method of ExpImageFilter in order to support a procedural API


    See:
     itk::simple::ExpImageFilter for the object oriented interface



    
### sitk.ExpImageFilter


    Computes the exponential function of each pixel.


    The computation is performed using std::exp(x).
    See:
     itk::simple::Exp for the procedural interface

     itk::ExpImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkExpImageFilter.h

    
### sitk.ExpNegative
    ExpNegative(Image image1) -> Image



    Computes the function exp(-K.x) for each input pixel.


    This function directly calls the execute method of ExpNegativeImageFilter in order to support a procedural API


    See:
     itk::simple::ExpNegativeImageFilter for the object oriented interface



    
### sitk.ExpNegativeImageFilter


    Computes the function exp(-K.x) for each input pixel.


    Every output pixel is equal to std::exp(-K.x ). where x is the
    intensity of the homologous input pixel, and K is a user-provided
    constant.
    See:
     itk::simple::ExpNegative for the procedural interface

     itk::ExpNegativeImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkExpNegativeImageFilter.h

    
### sitk.Expand
    Expand(Image image1, VectorUInt32 expandFactors, itk::simple::InterpolatorEnum interpolator) -> Image



    Expand the size of an image by an integer factor in each dimension.


    This function directly calls the execute method of ExpandImageFilter in order to support a procedural API


    See:
     itk::simple::ExpandImageFilter for the object oriented interface



    
### sitk.ExpandImageFilter


    Expand the size of an image by an integer factor in each dimension.


    ExpandImageFilter increases the size of an image by an integer factor in each dimension
    using a interpolation method. The output image size in each dimension
    is given by:

    OutputSize[j] = InputSize[j] * ExpandFactors[j]

    The output values are obtained by interpolating the input image. The
    default interpolation type used is the LinearInterpolateImageFunction . The user can specify a particular interpolation function via SetInterpolator() . Note that the input interpolator must derive from base class InterpolateImageFunction .

    This filter will produce an output with different pixel spacing that
    its input image such that:

    OutputSpacing[j] = InputSpacing[j] / ExpandFactors[j]

    The filter is templated over the input image type and the output image
    type.

    This filter is implemented as a multithreaded filter and supports
    streaming.


    WARNING:
    This filter only works for image with scalar pixel types. For vector
    images use VectorExpandImageFilter .
     This filter assumes that the input and output image has the same
    number of dimensions.


    See:
     InterpolateImageFunction

     LinearInterpolationImageFunction

     VectorExpandImageFilter

     itk::simple::Expand for the procedural interface

     itk::ExpandImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkExpandImageFilter.h

    
### sitk.Extract
    Extract(Image image1, VectorUInt32 size, VectorInt32 index, itk::simple::ExtractImageFilter::DirectionCollapseToStrategyType directionCollapseToStrategy) -> Image



    Decrease the image size by cropping the image to the selected region
    bounds.


    This function directly calls the execute method of ExtractImageFilter in order to support a procedural API


    See:
     itk::simple::ExtractImageFilter for the object oriented interface



    
### sitk.ExtractImageFilter


    Decrease the image size by cropping the image to the selected region
    bounds.


    ExtractImageFilter changes the image boundary of an image by removing pixels outside the
    target region. The target region must be specified.

    ExtractImageFilter also collapses dimensions so that the input image may have more
    dimensions than the output image (i.e. 4-D input image to a 3-D output
    image). To specify what dimensions to collapse, the ExtractionRegion
    must be specified. For any dimension dim where
    ExtractionRegion.Size[dim] = 0, that dimension is collapsed. The index
    to collapse on is specified by ExtractionRegion.Index[dim]. For
    example, we have a image 4D = a 4x4x4x4 image, and we want to get a 3D
    image, 3D = a 4x4x4 image, specified as [x,y,z,2] from 4D (i.e. the
    3rd "time" slice from 4D). The ExtractionRegion.Size = [4,4,4,0] and
    ExtractionRegion.Index = [0,0,0,2].

    The number of dimension in ExtractionRegion.Size and Index must = InputImageDimension. The number of non-zero dimensions in
    ExtractionRegion.Size must = OutputImageDimension.

    The output image produced by this filter will have the same origin as
    the input image, while the ImageRegion of the output image will start at the starting index value provided
    in the ExtractRegion parameter. If you are looking for a filter that
    will re-compute the origin of the output image, and provide an output
    image region whose index is set to zeros, then you may want to use the RegionOfInterestImageFilter . The output spacing is is simply the collapsed version of the input
    spacing.

    Determining the direction of the collapsed output image from an larger
    dimensional input space is an ill defined problem in general. It is
    required that the application developer select the desired
    transformation strategy for collapsing direction cosines. It is
    REQUIRED that a strategy be explicitly requested (i.e. there is no
    working default). Direction Collapsing Strategies: 1)
    DirectionCollapseToUnknown(); This is the default and the filter can
    not run when this is set. The reason is to explicitly force the
    application developer to define their desired behavior. 1)
    DirectionCollapseToIdentity(); Output has identity direction no matter
    what 2) DirectionCollapseToSubmatrix(); Output direction is the sub-
    matrix if it is positive definite, else throw an exception.

    This filter is implemented as a multithreaded filter. It provides a
    ThreadedGenerateData() method for its implementation.


    This filter is derived from InPlaceImageFilter . When the input to this filter matched the output requirested
    region, like with streaming filter for input, then setting this filter
    to run in-place will result in no copying of the bulk pixel data.

    See:
     CropImageFilter

     itk::simple::Extract for the procedural interface

     itk::ExtractImageFilter<InputImageType, typename InputImageType::template Rebind for the
    Doxygen on the original ITK class.


    C++ includes: sitkExtractImageFilter.h

    
### sitk.FFTConvolution
    FFTConvolution(Image image, Image kernelImage, bool normalize=False, itk::simple::FFTConvolutionImageFilter::BoundaryConditionType boundaryCondition, itk::simple::FFTConvolutionImageFilter::OutputRegionModeType outputRegionMode) -> Image



    Convolve a given image with an arbitrary image kernel using
    multiplication in the Fourier domain.


    This function directly calls the execute method of FFTConvolutionImageFilter in order to support a procedural API


    See:
     itk::simple::FFTConvolutionImageFilter for the object oriented interface



    
### sitk.FFTConvolutionImageFilter


    Convolve a given image with an arbitrary image kernel using
    multiplication in the Fourier domain.


    This filter produces output equivalent to the output of the ConvolutionImageFilter . However, it takes advantage of the convolution theorem to
    accelerate the convolution computation when the kernel is large.


    WARNING:
    This filter ignores the spacing, origin, and orientation of the kernel
    image and treats them as identical to those in the input image.
     This code was adapted from the Insight Journal contribution:

    "FFT Based Convolution" by Gaetan Lehmann https://hdl.handle.net/10380/3154


    See:
     ConvolutionImageFilter

     itk::simple::FFTConvolution for the procedural interface

     itk::FFTConvolutionImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkFFTConvolutionImageFilter.h

    
### sitk.FFTNormalizedCorrelation
    FFTNormalizedCorrelation(Image image1, Image image2, uint64_t requiredNumberOfOverlappingPixels=0) -> Image



    Calculate normalized cross correlation using FFTs.


    This function directly calls the execute method of FFTNormalizedCorrelationImageFilter in order to support a procedural API


    See:
     itk::simple::FFTNormalizedCorrelationImageFilter for the object oriented interface



    
### sitk.FFTNormalizedCorrelationImageFilter


    Calculate normalized cross correlation using FFTs.


    This filter calculates the normalized cross correlation (NCC) of two
    images using FFTs instead of spatial correlation. It is much faster
    than spatial correlation for reasonably large structuring elements.
    This filter is a subclass of the more general MaskedFFTNormalizedCorrelationImageFilter and operates by essentially setting the masks in that algorithm to
    images of ones. As described in detail in the references below, there
    is no computational overhead to utilizing the more general masked
    algorithm because the FFTs of the images of ones are still necessary
    for the computations.

    Inputs: Two images are required as inputs, fixedImage and movingImage.
    In the context of correlation, inputs are often defined as: "image"
    and "template". In this filter, the fixedImage plays the role of the
    image, and the movingImage plays the role of the template. However,
    this filter is capable of correlating any two images and is not
    restricted to small movingImages (templates).

    Optional parameters: The RequiredNumberOfOverlappingPixels enables the
    user to specify how many voxels of the two images must overlap; any
    location in the correlation map that results from fewer than this
    number of voxels will be set to zero. Larger values zero-out pixels on
    a larger border around the correlation image. Thus, larger values
    remove less stable computations but also limit the capture range. If
    RequiredNumberOfOverlappingPixels is set to 0, the default, no zeroing
    will take place.

    Image size: fixedImage and movingImage need not be the same size.
    Furthermore, whereas some algorithms require that the "template" be
    smaller than the "image" because of errors in the regions where the
    two are not fully overlapping, this filter has no such restriction.

    Image spacing: Since the computations are done in the pixel domain, all
    input images must have the same spacing.

    Outputs; The output is an image of RealPixelType that is the NCC of
    the two images and its values range from -1.0 to 1.0. The size of this
    NCC image is, by definition, size(fixedImage) + size(movingImage) - 1.

    Example filter usage:


    WARNING:
    The pixel type of the output image must be of real type (float or
    double). ConceptChecking is used to enforce the output pixel type. You
    will get a compilation error if the pixel type of the output image is
    not float or double.
     References: 1) D. Padfield. "Masked object registration in the
    Fourier domain." Transactions on Image Processing. 2) D. Padfield. "Masked FFT registration". In Proc.
    Computer Vision and Pattern Recognition, 2010.


    : Dirk Padfield, GE Global Research, padfield@research.ge.com

    See:
     itk::simple::FFTNormalizedCorrelation for the procedural interface

     itk::FFTNormalizedCorrelationImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkFFTNormalizedCorrelationImageFilter.h

    
### sitk.FFTPad
    FFTPad(Image image1, itk::simple::FFTPadImageFilter::BoundaryConditionType boundaryCondition, int sizeGreatestPrimeFactor=5) -> Image



    Pad an image to make it suitable for an FFT transformation.


    This function directly calls the execute method of FFTPadImageFilter in order to support a procedural API


    See:
     itk::simple::FFTPadImageFilter for the object oriented interface



    
### sitk.FFTPadImageFilter


    Pad an image to make it suitable for an FFT transformation.


    FFT filters usually requires a specific image size. The size is
    decomposed in several prime factors, and the filter only supports
    prime factors up to a maximum value. This filter automatically finds
    the greatest prime factor required by the available implementation and
    pads the input appropriately.

    This code was adapted from the Insight Journal contribution:

    "FFT Based Convolution" by Gaetan Lehmann https://hdl.handle.net/10380/3154


    Gaetan Lehmann

    See:
     FFTShiftImageFilter

     itk::simple::FFTPad for the procedural interface

     itk::FFTPadImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkFFTPadImageFilter.h

    
### sitk.FFTShift
    FFTShift(Image image1, bool inverse=False) -> Image



    Shift the zero-frequency components of a Fourier transform to the
    center of the image.


    This function directly calls the execute method of FFTShiftImageFilter in order to support a procedural API


    See:
     itk::simple::FFTShiftImageFilter for the object oriented interface



    
### sitk.FFTShiftImageFilter


    Shift the zero-frequency components of a Fourier transform to the
    center of the image.


    The Fourier transform produces an image where the zero frequency
    components are in the corner of the image, making it difficult to
    understand. This filter shifts the component to the center of the
    image.


    For images with an odd-sized dimension, applying this filter twice
    will not produce the same image as the original one without using
    SetInverse(true) on one (and only one) of the two filters.
    https://hdl.handle.net/1926/321


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.

    See:
     ForwardFFTImageFilter , InverseFFTImageFilter

     itk::simple::FFTShift for the procedural interface

     itk::FFTShiftImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkFFTShiftImageFilter.h

    
### sitk.FastApproximateRank
    FastApproximateRank(Image image1, double rank=0.5, VectorUInt32 radius) -> Image



    A separable rank filter.


    This function directly calls the execute method of FastApproximateRankImageFilter in order to support a procedural API


    See:
     itk::simple::FastApproximateRankImageFilter for the object oriented interface



    
### sitk.FastApproximateRankImageFilter


    A separable rank filter.


    Medians aren't separable, but if you want a large robust smoother to
    be relatively quick then it is worthwhile pretending that they are.

    This code was contributed in the Insight Journal paper: "Efficient
    implementation of kernel filtering" by Beare R., Lehmann G https://hdl.handle.net/1926/555 http://www.insight-journal.org/browse/publication/160


    Richard Beare

    See:
     itk::simple::FastApproximateRank for the procedural interface

     itk::FastApproximateRankImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkFastApproximateRankImageFilter.h

    
### sitk.FastMarching
    FastMarching(Image image1, VectorUIntList trialPoints, double normalizationFactor=1.0, double stoppingValue) -> Image



    Solve an Eikonal equation using Fast Marching.


    This function directly calls the execute method of FastMarchingImageFilter in order to support a procedural API


    See:
     itk::simple::FastMarchingImageFilter for the object oriented interface



    
### sitk.FastMarchingBase
    FastMarchingBase(Image imageA, VectorUIntList trialPoints, double normalizationFactor=1.0, double stoppingValue, itk::simple::FastMarchingBaseImageFilter::TopologyCheckType topologyCheck) -> Image



    itk::simple::FastMarchingBaseImageFilter Functional Interface

    This function directly calls the execute method of FastMarchingBaseImageFilter in order to support a fully functional API


    
### sitk.FastMarchingBaseImageFilter


    Apply the Fast Marching method to solve an Eikonal equation on an
    image.


    The speed function can be specified as a speed image or a speed
    constant. The speed image is set using the method SetInput(). If the
    speed image is ITK_NULLPTR, a constant speed function is used and is
    specified using method the SetSpeedConstant() .

    If the speed function is constant and of value one, fast marching
    results is an approximate distance function from the initial alive
    points.

    There are two ways to specify the output image information
    (LargestPossibleRegion, Spacing, Origin):


    it is copied directly from the input speed image

    it is specified by the user. Default values are used if the user does
    not specify all the information.
     The output information is computed as follows.

    If the speed image is ITK_NULLPTR or if the OverrideOutputInformation
    is set to true, the output information is set from user specified
    parameters. These parameters can be specified using methods


    FastMarchingImageFilterBase::SetOutputRegion() ,

    FastMarchingImageFilterBase::SetOutputSpacing() ,

    FastMarchingImageFilterBase::SetOutputDirection() ,

    FastMarchingImageFilterBase::SetOutputOrigin() .
     Else the output information is copied from the input speed image.

    Implementation of this class is based on Chapter 8 of "Level Set
    Methods and Fast Marching Methods", J.A. Sethian, Cambridge Press,
    Second edition, 1999.

    For an alternative implementation, see itk::FastMarchingImageFilter .

    TTraits

    traits


    See:
     FastMarchingImageFilter

     ImageFastMarchingTraits

     ImageFastMarchingTraits2

     itk::simple::FastMarchingBase for the procedural interface

     itk::FastMarchingImageFilterBase for the Doxygen on the original ITK class.


    C++ includes: sitkFastMarchingBaseImageFilter.h

    
### sitk.FastMarchingImageFilter


    Solve an Eikonal equation using Fast Marching.


    Fast marching solves an Eikonal equation where the speed is always
    non-negative and depends on the position only. Starting from an
    initial position on the front, fast marching systematically moves the
    front forward one grid point at a time.

    Updates are preformed using an entropy satisfy scheme where only
    "upwind" neighborhoods are used. This implementation of Fast
    Marching uses a std::priority_queue to locate the next proper grid
    position to update.

    Fast Marching sweeps through N grid points in (N log N) steps to
    obtain the arrival time value as the front propagates through the
    grid.

    Implementation of this class is based on Chapter 8 of "Level Set
    Methods and Fast Marching Methods", J.A. Sethian, Cambridge Press,
    Second edition, 1999.

    This class is templated over the level set image type and the speed
    image type. The initial front is specified by two containers: one
    containing the known points and one containing the trial points. Alive
    points are those that are already part of the object, and trial points
    are considered for inclusion. In order for the filter to evolve, at
    least some trial points must be specified. These can for instance be
    specified as the layer of pixels around the alive points.

    The speed function can be specified as a speed image or a speed
    constant. The speed image is set using the method SetInput() . If the
    speed image is ITK_NULLPTR, a constant speed function is used and is
    specified using method the SetSpeedConstant() .

    If the speed function is constant and of value one, fast marching
    results in an approximate distance function from the initial alive
    points. FastMarchingImageFilter is used in the ReinitializeLevelSetImageFilter object to create a signed distance function from the zero level set.

    The algorithm can be terminated early by setting an appropriate
    stopping value. The algorithm terminates when the current arrival time
    being processed is greater than the stopping value.

    There are two ways to specify the output image information (
    LargestPossibleRegion, Spacing, Origin): (a) it is copied directly
    from the input speed image or (b) it is specified by the user. Default
    values are used if the user does not specify all the information.

    The output information is computed as follows. If the speed image is
    ITK_NULLPTR or if the OverrideOutputInformation is set to true, the
    output information is set from user specified parameters. These
    parameters can be specified using methods SetOutputRegion() ,
    SetOutputSpacing() , SetOutputDirection() , and SetOutputOrigin() .
    Else if the speed image is not ITK_NULLPTR, the output information is
    copied from the input speed image.

    For an alternative implementation, see itk::FastMarchingImageFilter .

    Possible Improvements: In the current implementation,
    std::priority_queue only allows taking nodes out from the front and
    putting nodes in from the back. To update a value already on the heap,
    a new node is added to the heap. The defunct old node is left on the
    heap. When it is removed from the top, it will be recognized as
    invalid and not used. Future implementations can implement the heap in
    a different way allowing the values to be updated. This will generally
    require some sift-up and sift-down functions and an image of back-
    pointers going from the image to heap in order to locate the node
    which is to be updated.


    See:
     FastMarchingImageFilterBase

     LevelSetTypeDefault

     itk::simple::FastMarching for the procedural interface

     itk::FastMarchingImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkFastMarchingImageFilter.h

    
### sitk.FastMarchingUpwindGradient
    FastMarchingUpwindGradient(Image image1, VectorUIntList trialPoints, unsigned int numberOfTargets=0, VectorUIntList targetPoints, double targetOffset=1, double normalizationFactor=1.0) -> Image



    Generates the upwind gradient field of fast marching arrival times.


    This function directly calls the execute method of FastMarchingUpwindGradientImageFilter in order to support a procedural API


    See:
     itk::simple::FastMarchingUpwindGradientImageFilter for the object oriented interface



    
### sitk.FastMarchingUpwindGradientImageFilter


    Generates the upwind gradient field of fast marching arrival times.


    This filter adds some extra functionality to its base class. While the
    solution T(x) of the Eikonal equation is being generated by the base
    class with the fast marching method, the filter generates the upwind
    gradient vectors of T(x), storing them in an image.

    Since the Eikonal equation generates the arrival times of a wave
    travelling at a given speed, the generated gradient vectors can be
    interpreted as the slowness (1/velocity) vectors of the front (the
    quantity inside the modulus operator in the Eikonal equation).

    Gradient vectors are computed using upwind finite differences, that
    is, information only propagates from points where the wavefront has
    already passed. This is consistent with how the fast marching method
    works.

    One more extra feature is the possibility to define a set of Target
    points where the propagation stops. This can be used to avoid
    computing the Eikonal solution for the whole domain. The front can be
    stopped either when one Target point is reached or all Target points
    are reached. The propagation can stop after a time TargetOffset has
    passed since the stop condition is met. This way the solution is
    computed a bit downstream the Target points, so that the level sets of
    T(x) corresponding to the Target are smooth.

    For an alternative implementation, see itk::FastMarchingUpwindGradientImageFilterBase .


    Luca Antiga Ph.D. Biomedical Technologies Laboratory, Bioengineering
    Department, Mario Negri Institute, Italy.

    See:
     itk::simple::FastMarchingUpwindGradient for the procedural interface

     itk::FastMarchingUpwindGradientImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkFastMarchingUpwindGradientImageFilter.h

    
### sitk.FastSymmetricForcesDemonsRegistrationFilter


    Deformably register two images using a symmetric forces demons
    algorithm.


    This class was contributed by Tom Vercauteren, INRIA & Mauna Kea
    Technologies based on a variation of the DemonsRegistrationFilter .

    FastSymmetricForcesDemonsRegistrationFilter implements the demons deformable algorithm that register two images
    by computing the deformation field which will map a moving image onto
    a fixed image.

    A deformation field is represented as a image whose pixel type is some
    vector type with at least N elements, where N is the dimension of the
    fixed image. The vector type must support element access via operator
    []. It is assumed that the vector elements behave like floating point
    scalars.

    This class is templated over the fixed image type, moving image type
    and the deformation field type.

    The input fixed and moving images are set via methods SetFixedImage
    and SetMovingImage respectively. An initial deformation field maybe
    set via SetInitialDisplacementField or SetInput. If no initial field
    is set, a zero field is used as the initial condition.

    The output deformation field can be obtained via methods GetOutput or
    GetDisplacementField.

    This class make use of the finite difference solver hierarchy. Update
    for each iteration is computed in DemonsRegistrationFunction .


    Tom Vercauteren, INRIA & Mauna Kea Technologies
     This implementation was taken from the Insight Journal paper: https://hdl.handle.net/1926/510


    WARNING:
    This filter assumes that the fixed image type, moving image type and
    deformation field type all have the same number of dimensions.

    See:
     DemonsRegistrationFilter

     DemonsRegistrationFunction

     itk::FastSymmetricForcesDemonsRegistrationFilter for the Doxygen on the original ITK class.


    C++ includes: sitkFastSymmetricForcesDemonsRegistrationFilter.h

    
### sitk.Flip
    Flip(Image image1, VectorBool flipAxes, bool flipAboutOrigin=False) -> Image



    Flips an image across user specified axes.


    This function directly calls the execute method of FlipImageFilter in order to support a procedural API


    See:
     itk::simple::FlipImageFilter for the object oriented interface



    
### sitk.FlipImageFilter


    Flips an image across user specified axes.


    FlipImageFilter flips an image across user specified axes. The flip axes are set via
    method SetFlipAxes( array ) where the input is a
    FixedArray<bool,ImageDimension>. The image is flipped across axes for
    which array[i] is true.

    In terms of grid coordinates the image is flipped within the
    LargestPossibleRegion of the input image. As such, the
    LargestPossibleRegion of the output image is the same as the input.

    In terms of geometric coordinates, the output origin is such that the
    image is flipped with respect to the coordinate axes.
    See:
     itk::simple::Flip for the procedural interface

     itk::FlipImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkFlipImageFilter.h

    
### sitk.ForwardFFT
    ForwardFFT(Image image1) -> Image



    Base class for forward Fast Fourier Transform .


    This function directly calls the execute method of ForwardFFTImageFilter in order to support a procedural API


    See:
     itk::simple::ForwardFFTImageFilter for the object oriented interface



    
### sitk.ForwardFFTImageFilter


    Base class for forward Fast Fourier Transform .


    This is a base class for the "forward" or "direct" discrete
    Fourier Transform . This is an abstract base class: the actual implementation is
    provided by the best child class available on the system when the
    object is created via the object factory system.

    This class transforms a real input image into its full complex Fourier
    transform. The Fourier transform of a real input image has Hermitian
    symmetry: $ f(\mathbf{x}) = f^*(-\mathbf{x}) $ . That is, when the result of the transform is split in half along
    the x-dimension, the values in the second half of the transform are
    the complex conjugates of values in the first half reflected about the
    center of the image in each dimension.

    This filter works only for real single-component input image types.


    See:
     InverseFFTImageFilter , FFTComplexToComplexImageFilter

     itk::simple::ForwardFFT for the procedural interface

     itk::ForwardFFTImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkForwardFFTImageFilter.h

    
### sitk.GaborImageSource


    Generate an n-dimensional image of a Gabor filter.


    GaborImageSource generates an image of either the real (i.e. symmetric) or complex
    (i.e. antisymmetric) part of the Gabor filter with the orientation
    directed along the x-axis. The GaborKernelFunction is used to evaluate the contribution along the x-axis whereas a non-
    normalized 1-D Gaussian envelope provides the contribution in each of
    the remaining N dimensions. Orientation can be manipulated via the Transform classes of the toolkit.

    The output image may be of any dimension.

    This implementation was contributed as a paper to the Insight Journal https://hdl.handle.net/1926/500
    See:
     itk::simple::GaborSource for the procedural interface

     itk::GaborImageSource for the Doxygen on the original ITK class.


    C++ includes: sitkGaborImageSource.h

    
### sitk.GaborSource
    GaborSource(itk::simple::PixelIDValueEnum outputPixelType, VectorUInt32 size, VectorDouble sigma, VectorDouble mean, double frequency=0.4, VectorDouble origin, VectorDouble spacing, VectorDouble direction) -> Image



    Generate an n-dimensional image of a Gabor filter.


    This function directly calls the execute method of GaborImageSource in order to support a procedural API


    See:
     itk::simple::GaborImageSource for the object oriented interface



    
### sitk.GaussianImageSource


    Generate an n-dimensional image of a Gaussian.


    GaussianImageSource generates an image of a Gaussian. m_Normalized determines whether or
    not the Gaussian is normalized (whether or not the sum over infinite
    space is 1.0) When creating an image, it is preferable to not
    normalize the Gaussian m_Scale scales the output of the Gaussian to
    span a range larger than 0->1, and is typically set to the maximum
    value of the output data type (for instance, 255 for uchars)

    The output image may be of any dimension.
    See:
     itk::simple::GaussianSource for the procedural interface

     itk::GaussianImageSource for the Doxygen on the original ITK class.


    C++ includes: sitkGaussianImageSource.h

    
### sitk.GaussianSource
    GaussianSource(itk::simple::PixelIDValueEnum outputPixelType, VectorUInt32 size, VectorDouble sigma, VectorDouble mean, double scale=255, VectorDouble origin, VectorDouble spacing, VectorDouble direction, bool normalized=False) -> Image



    Generate an n-dimensional image of a Gaussian.


    This function directly calls the execute method of GaussianImageSource in order to support a procedural API


    See:
     itk::simple::GaussianImageSource for the object oriented interface



    
### sitk.GeodesicActiveContourLevelSet
    GeodesicActiveContourLevelSet(Image image1, Image image2, double maximumRMSError=0.01, double propagationScaling=1.0, double curvatureScaling=1.0, double advectionScaling=1.0, uint32_t numberOfIterations=1000, bool reverseExpansionDirection=False) -> Image



    Segments structures in images based on a user supplied edge potential
    map.


    This function directly calls the execute method of GeodesicActiveContourLevelSetImageFilter in order to support a procedural API


    See:
     itk::simple::GeodesicActiveContourLevelSetImageFilter for the object oriented interface



    
### sitk.GeodesicActiveContourLevelSetImageFilter


    Segments structures in images based on a user supplied edge potential
    map.


    IMPORTANT
    The SegmentationLevelSetImageFilter class and the GeodesicActiveContourLevelSetFunction class contain additional information necessary to gain full
    understanding of how to use this filter.
    OVERVIEW
    This class is a level set method segmentation filter. An initial
    contour is propagated outwards (or inwards) until it ''sticks'' to the
    shape boundaries. This is done by using a level set speed function
    based on a user supplied edge potential map.
    INPUTS
    This filter requires two inputs. The first input is a initial level
    set. The initial level set is a real image which contains the initial
    contour/surface as the zero level set. For example, a signed distance
    function from the initial contour/surface is typically used. Unlike
    the simpler ShapeDetectionLevelSetImageFilter the initial contour does not have to lie wholly within the shape to
    be segmented. The initial contour is allow to overlap the shape
    boundary. The extra advection term in the update equation behaves like
    a doublet and attracts the contour to the boundary. This approach for
    segmentation follows that of Caselles et al (1997).

    The second input is the feature image. For this filter, this is the
    edge potential map. General characteristics of an edge potential map
    is that it has values close to zero in regions near the edges and
    values close to one inside the shape itself. Typically, the edge
    potential map is compute from the image gradient, for example:
    \[ g(I) = 1 / ( 1 + | (\nabla * G)(I)| ) \] \[ g(I) = \exp^{-|(\nabla * G)(I)|} \]

    where $ I $ is image intensity and $ (\nabla * G) $ is the derivative of Gaussian operator.


    See SegmentationLevelSetImageFilter and SparseFieldLevelSetImageFilter for more information on Inputs.
    PARAMETERS
    The PropagationScaling parameter can be used to switch from
    propagation outwards (POSITIVE scaling parameter) versus propagating
    inwards (NEGATIVE scaling parameter).
     This implementation allows the user to set the weights between the
    propagation, advection and curvature term using methods SetPropagationScaling() , SetAdvectionScaling() , SetCurvatureScaling() . In general, the larger the CurvatureScaling, the smoother the
    resulting contour. To follow the implementation in Caselles et al
    paper, set the PropagationScaling to $ c $ (the inflation or ballon force) and AdvectionScaling and
    CurvatureScaling both to 1.0.

    OUTPUTS
    The filter outputs a single, scalar, real-valued image. Negative
    values in the output image represent the inside of the segmented
    region and positive values in the image represent the outside of the
    segmented region. The zero crossings of the image correspond to the
    position of the propagating front.

    See SparseFieldLevelSetImageFilter and SegmentationLevelSetImageFilter for more information.
    REFERENCES

    "Geodesic Active Contours", V. Caselles, R. Kimmel and G. Sapiro.
    International Journal on Computer Vision, Vol 22, No. 1, pp 61-97,
    1997

    See:
     SegmentationLevelSetImageFilter

     GeodesicActiveContourLevelSetFunction

     SparseFieldLevelSetImageFilter

     itk::simple::GeodesicActiveContourLevelSet for the procedural interface

     itk::GeodesicActiveContourLevelSetImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkGeodesicActiveContourLevelSetImageFilter.h

    
### sitk.GetArrayFromImageGet a NumPy ndarray from a SimpleITK Image.

    This is a deep copy of the image buffer and is completely safe and without potential side effects.
    
### sitk.GetArrayViewFromImageGet a NumPy ndarray view of a SimpleITK Image.

    Returns a Numpy ndarray object as a "view" of the SimpleITK's Image buffer. This reduces pixel buffer copies, but requires that the SimpleITK image object is kept around while the buffer is being used.


    
### sitk.GetImageFromArrayGet a SimpleITK Image from a numpy array. If isVector is True, then the Image will have a Vector pixel type, and the last dimension of the array will be considered the component index. By default when isVector is None, 4D images are automatically considered 3D vector images.
### sitk.GetPixelIDValueAsString
    GetPixelIDValueAsString(itk::simple::PixelIDValueEnum type) -> std::string const



    
### sitk.GetPixelIDValueFromString
    GetPixelIDValueFromString(std::string const & enumString) -> itk::simple::PixelIDValueType



    Function mapping enumeration names in std::string to values.


    This function is intended for use by the R bindings. R stores the
    enumeration values using the names : "sitkUnkown", "sitkUInt8",
    etc from PixelIDValueEnum above. This function is used to provide the
    integer values using calls like:

    val = GetPixelIDValueFromString("sitkInt32")

    If the pixel type has not been instantiated then the sitkUnknown value
    (-1) will be returned. If the pixel type string is not recognised
    (i.e. is not in the set of tested names) then the return value is -99.
    The idea is to provide a warning (via the R package) if this function
    needs to be updated to match changes to PixelIDValueEnum - i.e. if a
    new pixel type is added.


    
### sitk.Gradient
    Gradient(Image image1, bool useImageSpacing=True, bool useImageDirection=False) -> Image



    Computes the gradient of an image using directional derivatives.


    This function directly calls the execute method of GradientImageFilter in order to support a procedural API


    See:
     itk::simple::GradientImageFilter for the object oriented interface



    
### sitk.GradientAnisotropicDiffusion
    GradientAnisotropicDiffusion(Image image1, double timeStep=0.125, double conductanceParameter=3, unsigned int conductanceScalingUpdateInterval=1, uint32_t numberOfIterations=5) -> Image



    itk::simple::GradientAnisotropicDiffusionImageFilter Procedural Interface


    This function directly calls the execute method of GradientAnisotropicDiffusionImageFilter in order to support a procedural API


    See:
     itk::simple::GradientAnisotropicDiffusionImageFilter for the object oriented interface



    
### sitk.GradientAnisotropicDiffusionImageFilter


    This filter performs anisotropic diffusion on a scalar itk::Image using the classic Perona-Malik, gradient magnitude based equation
    implemented in itkGradientNDAnisotropicDiffusionFunction. For detailed
    information on anisotropic diffusion, see
    itkAnisotropicDiffusionFunction and
    itkGradientNDAnisotropicDiffusionFunction.

    Inputs and Outputs
    The input to this filter should be a scalar itk::Image of any dimensionality. The output image will be a diffused copy of
    the input.
    Parameters
    Please see the description of parameters given in
    itkAnisotropicDiffusionImageFilter.

    See:
     AnisotropicDiffusionImageFilter

     AnisotropicDiffusionFunction

     GradientAnisotropicDiffusionFunction

     itk::simple::GradientAnisotropicDiffusion for the procedural interface

     itk::GradientAnisotropicDiffusionImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkGradientAnisotropicDiffusionImageFilter.h

    
### sitk.GradientImageFilter


    Computes the gradient of an image using directional derivatives.


    Computes the gradient of an image using directional derivatives. The
    directional derivative at each pixel location is computed by
    convolution with a first-order derivative operator.

    The second template parameter defines the value type used in the
    derivative operator (defaults to float). The third template parameter
    defines the value type used for output image (defaults to float). The
    output image is defined as a covariant vector image whose value type
    is specified as this third template parameter.


    See:
     Image

     Neighborhood

     NeighborhoodOperator

     NeighborhoodIterator

     itk::simple::Gradient for the procedural interface

     itk::GradientImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkGradientImageFilter.h

    
### sitk.GradientMagnitude
    GradientMagnitude(Image image1, bool useImageSpacing=True) -> Image



    Computes the gradient magnitude of an image region at each pixel.


    This function directly calls the execute method of GradientMagnitudeImageFilter in order to support a procedural API


    See:
     itk::simple::GradientMagnitudeImageFilter for the object oriented interface



    
### sitk.GradientMagnitudeImageFilter


    Computes the gradient magnitude of an image region at each pixel.



    See:
     Image

     Neighborhood

     NeighborhoodOperator

     NeighborhoodIterator

     itk::simple::GradientMagnitude for the procedural interface

     itk::GradientMagnitudeImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkGradientMagnitudeImageFilter.h

    
### sitk.GradientMagnitudeRecursiveGaussian
    GradientMagnitudeRecursiveGaussian(Image image1, double sigma=1.0, bool normalizeAcrossScale=False) -> Image



    Computes the Magnitude of the Gradient of an image by convolution with
    the first derivative of a Gaussian.


    This function directly calls the execute method of GradientMagnitudeRecursiveGaussianImageFilter in order to support a procedural API


    See:
     itk::simple::GradientMagnitudeRecursiveGaussianImageFilter for the object oriented interface



    
### sitk.GradientMagnitudeRecursiveGaussianImageFilter


    Computes the Magnitude of the Gradient of an image by convolution with
    the first derivative of a Gaussian.


    This filter is implemented using the recursive gaussian filters
    See:
     itk::simple::GradientMagnitudeRecursiveGaussian for the procedural interface

     itk::GradientMagnitudeRecursiveGaussianImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkGradientMagnitudeRecursiveGaussianImageFilter.h

    
### sitk.GradientRecursiveGaussian
    GradientRecursiveGaussian(Image image1, double sigma=1.0, bool normalizeAcrossScale=False, bool useImageDirection=False) -> Image



    Computes the gradient of an image by convolution with the first
    derivative of a Gaussian.


    This function directly calls the execute method of GradientRecursiveGaussianImageFilter in order to support a procedural API


    See:
     itk::simple::GradientRecursiveGaussianImageFilter for the object oriented interface



    
### sitk.GradientRecursiveGaussianImageFilter


    Computes the gradient of an image by convolution with the first
    derivative of a Gaussian.


    This filter is implemented using the recursive gaussian filters.

    This filter supports both scalar and vector pixel types within the
    input image, including VectorImage type.
    See:
     itk::simple::GradientRecursiveGaussian for the procedural interface

     itk::GradientRecursiveGaussianImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkGradientRecursiveGaussianImageFilter.h

    
### sitk.GrayscaleConnectedClosing
    GrayscaleConnectedClosing(Image image1, VectorUInt32 seed, bool fullyConnected=False) -> Image



    Enhance pixels associated with a dark object (identified by a seed
    pixel) where the dark object is surrounded by a brigher object.


    This function directly calls the execute method of GrayscaleConnectedClosingImageFilter in order to support a procedural API


    See:
     itk::simple::GrayscaleConnectedClosingImageFilter for the object oriented interface



    
### sitk.GrayscaleConnectedClosingImageFilter


    Enhance pixels associated with a dark object (identified by a seed
    pixel) where the dark object is surrounded by a brigher object.


    GrayscaleConnectedClosingImagefilter is useful for enhancing dark
    objects that are surrounded by bright borders. This filter makes it
    easier to threshold the image and extract just the object of interest.

    Geodesic morphology and the connected closing algorithm are described
    in Chapter 6 of Pierre Soille's book "Morphological Image Analysis:
    Principles and Applications", Second Edition, Springer, 2003.


    See:
     GrayscaleGeodesicDilateImageFilter

     MorphologyImageFilter , GrayscaleDilateImageFilter , GrayscaleFunctionDilateImageFilter , BinaryDilateImageFilter

     itk::simple::GrayscaleConnectedClosing for the procedural interface

     itk::GrayscaleConnectedClosingImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkGrayscaleConnectedClosingImageFilter.h

    
### sitk.GrayscaleConnectedOpening
    GrayscaleConnectedOpening(Image image1, VectorUInt32 seed, bool fullyConnected=False) -> Image



    Enhance pixels associated with a bright object (identified by a seed
    pixel) where the bright object is surrounded by a darker object.


    This function directly calls the execute method of GrayscaleConnectedOpeningImageFilter in order to support a procedural API


    See:
     itk::simple::GrayscaleConnectedOpeningImageFilter for the object oriented interface



    
### sitk.GrayscaleConnectedOpeningImageFilter


    Enhance pixels associated with a bright object (identified by a seed
    pixel) where the bright object is surrounded by a darker object.


    GrayscaleConnectedOpeningImagefilter is useful for enhancing bright
    objects that are surrounded by dark borders. This filter makes it
    easier to threshold the image and extract just the object of interest.

    Geodesic morphology and the connected opening algorithm is described
    in Chapter 6 of Pierre Soille's book "Morphological Image Analysis:
    Principles and Applications", Second Edition, Springer, 2003.


    See:
     GrayscaleGeodesicDilateImageFilter

     MorphologyImageFilter , GrayscaleDilateImageFilter , GrayscaleFunctionDilateImageFilter , BinaryDilateImageFilter

     itk::simple::GrayscaleConnectedOpening for the procedural interface

     itk::GrayscaleConnectedOpeningImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkGrayscaleConnectedOpeningImageFilter.h

    
### sitk.GrayscaleDilate
    GrayscaleDilate(Image arg1, uint32_t radius=1, itk::simple::KernelEnum kernel) -> Image
    GrayscaleDilate(Image arg1, VectorUInt32 vectorRadius, itk::simple::KernelEnum kernel) -> Image



    itk::simple::GrayscaleDilateImageFilter Functional Interface

    This function directly calls the execute method of GrayscaleDilateImageFilter in order to support a fully functional API


    
### sitk.GrayscaleDilateImageFilter


    Grayscale dilation of an image.


    Dilate an image using grayscale morphology. Dilation takes the maximum
    of all the pixels identified by the structuring element.

    The structuring element is assumed to be composed of binary values
    (zero or one). Only elements of the structuring element having values
    > 0 are candidates for affecting the center pixel.


    See:
     MorphologyImageFilter , GrayscaleFunctionDilateImageFilter , BinaryDilateImageFilter

     itk::simple::GrayscaleDilate for the procedural interface

     itk::GrayscaleDilateImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkGrayscaleDilateImageFilter.h

    
### sitk.GrayscaleErode
    GrayscaleErode(Image arg1, uint32_t radius=1, itk::simple::KernelEnum kernel) -> Image
    GrayscaleErode(Image arg1, VectorUInt32 vectorRadius, itk::simple::KernelEnum kernel) -> Image



    itk::simple::GrayscaleErodeImageFilter Functional Interface

    This function directly calls the execute method of GrayscaleErodeImageFilter in order to support a fully functional API


    
### sitk.GrayscaleErodeImageFilter


    Grayscale erosion of an image.


    Erode an image using grayscale morphology. Erosion takes the maximum
    of all the pixels identified by the structuring element.

    The structuring element is assumed to be composed of binary values
    (zero or one). Only elements of the structuring element having values
    > 0 are candidates for affecting the center pixel.


    See:
     MorphologyImageFilter , GrayscaleFunctionErodeImageFilter , BinaryErodeImageFilter

     itk::simple::GrayscaleErode for the procedural interface

     itk::GrayscaleErodeImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkGrayscaleErodeImageFilter.h

    
### sitk.GrayscaleFillhole
    GrayscaleFillhole(Image image1, bool fullyConnected=False) -> Image



    Remove local minima not connected to the boundary of the image.


    This function directly calls the execute method of GrayscaleFillholeImageFilter in order to support a procedural API


    See:
     itk::simple::GrayscaleFillholeImageFilter for the object oriented interface



    
### sitk.GrayscaleFillholeImageFilter


    Remove local minima not connected to the boundary of the image.


    GrayscaleFillholeImageFilter fills holes in a grayscale image. Holes are local minima in the
    grayscale topography that are not connected to boundaries of the
    image. Gray level values adjacent to a hole are extrapolated across
    the hole.

    This filter is used to smooth over local minima without affecting the
    values of local maxima. If you take the difference between the output
    of this filter and the original image (and perhaps threshold the
    difference above a small value), you'll obtain a map of the local
    minima.

    This filter uses the ReconstructionByErosionImageFilter . It provides its own input as the "mask" input to the geodesic
    erosion. The "marker" image for the geodesic erosion is constructed
    such that boundary pixels match the boundary pixels of the input image
    and the interior pixels are set to the maximum pixel value in the
    input image.

    Geodesic morphology and the Fillhole algorithm is described in Chapter
    6 of Pierre Soille's book "Morphological Image Analysis: Principles
    and Applications", Second Edition, Springer, 2003.


    See:
     ReconstructionByErosionImageFilter

     MorphologyImageFilter , GrayscaleErodeImageFilter , GrayscaleFunctionErodeImageFilter , BinaryErodeImageFilter

     itk::simple::GrayscaleFillhole for the procedural interface

     itk::GrayscaleFillholeImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkGrayscaleFillholeImageFilter.h

    
### sitk.GrayscaleGeodesicDilate
    GrayscaleGeodesicDilate(Image image1, Image image2, bool runOneIteration=False, bool fullyConnected=False) -> Image



    geodesic gray scale dilation of an image


    This function directly calls the execute method of GrayscaleGeodesicDilateImageFilter in order to support a procedural API


    See:
     itk::simple::GrayscaleGeodesicDilateImageFilter for the object oriented interface



    
### sitk.GrayscaleGeodesicDilateImageFilter


    geodesic gray scale dilation of an image


    Geodesic dilation operates on a "marker" image and a "mask" image.
    The marker image is dilated using an elementary structuring element
    (neighborhood of radius one using only the face connected neighbors).
    The resulting image is then compared with the mask image. The output
    image is the pixelwise minimum of the dilated marker image and the
    mask image.

    Geodesic dilation is run either one iteration or until convergence. In
    the convergence case, the filter is equivalent to "reconstruction by
    dilation". This filter is implemented to handle both scenarios. The
    one iteration case is multi-threaded. The convergence case is
    delegated to another instance of the same filter (but configured to
    run a single iteration).

    The marker image must be less than or equal to the mask image (on a
    pixel by pixel basis).

    Geodesic morphology is described in Chapter 6 of Pierre Soille's book
    "Morphological Image Analysis: Principles and Applications", Second
    Edition, Springer, 2003.

    A noniterative version of this algorithm can be found in the ReconstructionByDilationImageFilter . This noniterative solution is much faster than the implementation
    provided here. All ITK filters that previously used
    GrayscaleGeodesicDiliateImageFilter as part of their implementation
    have been converted to use the ReconstructionByDilationImageFilter . The GrayscaleGeodesicDilateImageFilter is maintained for backward compatibility.


    See:
     MorphologyImageFilter , GrayscaleDilateImageFilter , GrayscaleFunctionDilateImageFilter , BinaryDilateImageFilter , ReconstructionByDilationImageFilter

     itk::simple::GrayscaleGeodesicDilate for the procedural interface

     itk::GrayscaleGeodesicDilateImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkGrayscaleGeodesicDilateImageFilter.h

    
### sitk.GrayscaleGeodesicErode
    GrayscaleGeodesicErode(Image image1, Image image2, bool runOneIteration=False, bool fullyConnected=False) -> Image



    geodesic gray scale erosion of an image


    This function directly calls the execute method of GrayscaleGeodesicErodeImageFilter in order to support a procedural API


    See:
     itk::simple::GrayscaleGeodesicErodeImageFilter for the object oriented interface



    
### sitk.GrayscaleGeodesicErodeImageFilter


    geodesic gray scale erosion of an image


    Geodesic erosion operates on a "marker" image and a "mask" image.
    The marker image is eroded using an elementary structuring element
    (neighborhood of radius one using only the face connected neighbors).
    The resulting image is then compared with the mask image. The output
    image is the pixelwise maximum of the eroded marker image and the mask
    image.

    Geodesic erosion is run either one iteration or until convergence. In
    the convergence case, the filter is equivalent to "reconstruction by
    erosion". This filter is implemented to handle both scenarios. The
    one iteration case is multi-threaded. The convergence case is
    delegated to another instance of the same filter (but configured to
    run a single iteration).

    The marker image must be greater than or equal to the mask image (on a
    pixel by pixel basis).

    Geodesic morphology is described in Chapter 6 of Pierre Soille's book
    "Morphological Image Analysis: Principles and Applications", Second
    Edition, Springer, 2003.

    A noniterative version of this algorithm can be found in the ReconstructionByErosionImageFilter . This noniterative solution is much faster than the implementation
    provided here. All ITK filters that previously used GrayscaleGeodesicErodeImageFilter as part of their implementation have been converted to use the ReconstructionByErosionImageFilter . The GrayscaleGeodesicErodeImageFilter is maintained for backward compatibility.


    See:
     MorphologyImageFilter , GrayscaleErodeImageFilter , GrayscaleFunctionErodeImageFilter , BinaryErodeImageFilter , ReconstructionByErosionImageFilter

     itk::simple::GrayscaleGeodesicErode for the procedural interface

     itk::GrayscaleGeodesicErodeImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkGrayscaleGeodesicErodeImageFilter.h

    
### sitk.GrayscaleGrindPeak
    GrayscaleGrindPeak(Image image1, bool fullyConnected=False) -> Image



    Remove local maxima not connected to the boundary of the image.


    This function directly calls the execute method of GrayscaleGrindPeakImageFilter in order to support a procedural API


    See:
     itk::simple::GrayscaleGrindPeakImageFilter for the object oriented interface



    
### sitk.GrayscaleGrindPeakImageFilter


    Remove local maxima not connected to the boundary of the image.


    GrayscaleGrindPeakImageFilter removes peaks in a grayscale image. Peaks are local maxima in the
    grayscale topography that are not connected to boundaries of the
    image. Gray level values adjacent to a peak are extrapolated through
    the peak.

    This filter is used to smooth over local maxima without affecting the
    values of local minima. If you take the difference between the output
    of this filter and the original image (and perhaps threshold the
    difference above a small value), you'll obtain a map of the local
    maxima.

    This filter uses the GrayscaleGeodesicDilateImageFilter . It provides its own input as the "mask" input to the geodesic
    erosion. The "marker" image for the geodesic erosion is constructed
    such that boundary pixels match the boundary pixels of the input image
    and the interior pixels are set to the minimum pixel value in the
    input image.

    This filter is the dual to the GrayscaleFillholeImageFilter which implements the Fillhole algorithm. Since it is a dual, it is
    somewhat superfluous but is provided as a convenience.

    Geodesic morphology and the Fillhole algorithm is described in Chapter
    6 of Pierre Soille's book "Morphological Image Analysis: Principles
    and Applications", Second Edition, Springer, 2003.


    See:
     GrayscaleGeodesicDilateImageFilter

     MorphologyImageFilter , GrayscaleDilateImageFilter , GrayscaleFunctionDilateImageFilter , BinaryDilateImageFilter

     itk::simple::GrayscaleGrindPeak for the procedural interface

     itk::GrayscaleGrindPeakImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkGrayscaleGrindPeakImageFilter.h

    
### sitk.GrayscaleMorphologicalClosing
    GrayscaleMorphologicalClosing(Image arg1, uint32_t radius=1, itk::simple::KernelEnum kernel, bool safeBorder=True) -> Image
    GrayscaleMorphologicalClosing(Image arg1, VectorUInt32 vectorRadius, itk::simple::KernelEnum kernel, bool safeBorder=True) -> Image



    itk::simple::GrayscaleMorphologicalClosingImageFilter Functional Interface

    This function directly calls the execute method of GrayscaleMorphologicalClosingImageFilter in order to support a fully functional API


    
### sitk.GrayscaleMorphologicalClosingImageFilter


    gray scale dilation of an image


    Erode an image using grayscale morphology. Dilation takes the maximum
    of all the pixels identified by the structuring element.

    The structuring element is assumed to be composed of binary values
    (zero or one). Only elements of the structuring element having values
    > 0 are candidates for affecting the center pixel.


    See:
     MorphologyImageFilter , GrayscaleFunctionErodeImageFilter , BinaryErodeImageFilter

     itk::simple::GrayscaleMorphologicalClosing for the procedural interface

     itk::GrayscaleMorphologicalClosingImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkGrayscaleMorphologicalClosingImageFilter.h

    
### sitk.GrayscaleMorphologicalOpening
    GrayscaleMorphologicalOpening(Image arg1, uint32_t radius=1, itk::simple::KernelEnum kernel, bool safeBorder=True) -> Image
    GrayscaleMorphologicalOpening(Image arg1, VectorUInt32 vectorRadius, itk::simple::KernelEnum kernel, bool safeBorder=True) -> Image



    itk::simple::GrayscaleMorphologicalOpeningImageFilter Functional Interface

    This function directly calls the execute method of GrayscaleMorphologicalOpeningImageFilter in order to support a fully functional API


    
### sitk.GrayscaleMorphologicalOpeningImageFilter


    gray scale dilation of an image


    Dilate an image using grayscale morphology. Dilation takes the maximum
    of all the pixels identified by the structuring element.

    The structuring element is assumed to be composed of binary values
    (zero or one). Only elements of the structuring element having values
    > 0 are candidates for affecting the center pixel.


    See:
     MorphologyImageFilter , GrayscaleFunctionDilateImageFilter , BinaryDilateImageFilter

     itk::simple::GrayscaleMorphologicalOpening for the procedural interface

     itk::GrayscaleMorphologicalOpeningImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkGrayscaleMorphologicalOpeningImageFilter.h

    
### sitk.Greater
    Greater(Image image1, Image image2, uint8_t backgroundValue=0, uint8_t foregroundValue=1) -> Image
    Greater(Image image1, double constant, uint8_t backgroundValue=0, uint8_t foregroundValue=1) -> Image
    Greater(double constant, Image image2, uint8_t backgroundValue=0, uint8_t foregroundValue=1) -> Image



    
### sitk.GreaterEqual
    GreaterEqual(Image image1, Image image2, uint8_t backgroundValue=0, uint8_t foregroundValue=1) -> Image
    GreaterEqual(Image image1, double constant, uint8_t backgroundValue=0, uint8_t foregroundValue=1) -> Image
    GreaterEqual(double constant, Image image2, uint8_t backgroundValue=0, uint8_t foregroundValue=1) -> Image



    
### sitk.GreaterEqualImageFilter


    Implements pixel-wise generic operation of two images, or of an image
    and a constant.


    This class is parameterized over the types of the two input images and
    the type of the output image. It is also parameterized by the
    operation to be applied. A Functor style is used.

    The constant must be of the same type than the pixel type of the
    corresponding image. It is wrapped in a SimpleDataObjectDecorator so it can be updated through the pipeline. The SetConstant() and
    GetConstant() methods are provided as shortcuts to set or get the
    constant value without manipulating the decorator.


    See:
     UnaryFunctorImageFilter TernaryFunctorImageFilter

     itk::simple::GreaterEqual for the procedural interface

     itk::BinaryFunctorImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkGreaterEqualImageFilter.h

    
### sitk.GreaterImageFilter


    Implements pixel-wise generic operation of two images, or of an image
    and a constant.


    This class is parameterized over the types of the two input images and
    the type of the output image. It is also parameterized by the
    operation to be applied. A Functor style is used.

    The constant must be of the same type than the pixel type of the
    corresponding image. It is wrapped in a SimpleDataObjectDecorator so it can be updated through the pipeline. The SetConstant() and
    GetConstant() methods are provided as shortcuts to set or get the
    constant value without manipulating the decorator.


    See:
     UnaryFunctorImageFilter TernaryFunctorImageFilter

     itk::simple::Greater for the procedural interface

     itk::BinaryFunctorImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkGreaterImageFilter.h

    
### sitk.GridImageSource


    Generate an n-dimensional image of a grid.


    GridImageSource generates an image of a grid. From the abstract... "Certain classes
    of images find disparate use amongst members of the ITK community for
    such purposes as visualization, simulation, testing, etc. Currently
    there exists two derived classes from the ImageSource class used for
    generating specific images for various applications, viz.
    RandomImageSource and GaussianImageSource . We propose to add to this
    set with the class GridImageSource which, obviously enough, produces a
    grid image. Such images are useful for visualizing deformation when
    used in conjunction with the WarpImageFilter , simulating magnetic
    resonance tagging images, or creating optical illusions with which to
    amaze your friends."

    The output image may be of any dimension.


    Tustison N., Avants B., Gee J. University of Pennsylvania
     This implementation was taken from the Insight Journal paper: https://hdl.handle.net/1926/475
    See:
     itk::simple::GridSource for the procedural interface

     itk::GridImageSource for the Doxygen on the original ITK class.


    C++ includes: sitkGridImageSource.h

    
### sitk.GridSource
    GridSource(itk::simple::PixelIDValueEnum outputPixelType, VectorUInt32 size, VectorDouble sigma, VectorDouble gridSpacing, VectorDouble gridOffset, double scale=255.0, VectorDouble origin, VectorDouble spacing, VectorDouble direction) -> Image



    Generate an n-dimensional image of a grid.


    This function directly calls the execute method of GridImageSource in order to support a procedural API


    See:
     itk::simple::GridImageSource for the object oriented interface



    
### sitk.HAVE_NUMPYbool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.
### sitk.HConcave
    HConcave(Image image1, double height=2.0, bool fullyConnected=False) -> Image



    Identify local minima whose depth below the baseline is greater than
    h.


    This function directly calls the execute method of HConcaveImageFilter in order to support a procedural API


    See:
     itk::simple::HConcaveImageFilter for the object oriented interface



    
### sitk.HConcaveImageFilter


    Identify local minima whose depth below the baseline is greater than
    h.


    HConcaveImageFilter extract local minima that are more than h intensity units below the
    (local) background. This has the effect of extracting objects that are
    darker than the background by at least h intensity units.

    This filter uses the HMinimaImageFilter .

    Geodesic morphology and the H-Convex algorithm is described in Chapter
    6 of Pierre Soille's book "Morphological Image Analysis: Principles
    and Applications", Second Edition, Springer, 2003.


    See:
     GrayscaleGeodesicDilateImageFilter , HMaximaImageFilter ,

     MorphologyImageFilter , GrayscaleDilateImageFilter , GrayscaleFunctionDilateImageFilter , BinaryDilateImageFilter

     itk::simple::HConcave for the procedural interface

     itk::HConcaveImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkHConcaveImageFilter.h

    
### sitk.HConvex
    HConvex(Image image1, double height=2.0, bool fullyConnected=False) -> Image



    Identify local maxima whose height above the baseline is greater than
    h.


    This function directly calls the execute method of HConvexImageFilter in order to support a procedural API


    See:
     itk::simple::HConvexImageFilter for the object oriented interface



    
### sitk.HConvexImageFilter


    Identify local maxima whose height above the baseline is greater than
    h.


    HConvexImageFilter extract local maxima that are more than h intensity units above the
    (local) background. This has the effect of extracting objects that are
    brighter than background by at least h intensity units.

    This filter uses the HMaximaImageFilter .

    Geodesic morphology and the H-Convex algorithm is described in Chapter
    6 of Pierre Soille's book "Morphological Image Analysis: Principles
    and Applications", Second Edition, Springer, 2003.


    See:
     GrayscaleGeodesicDilateImageFilter , HMinimaImageFilter

     MorphologyImageFilter , GrayscaleDilateImageFilter , GrayscaleFunctionDilateImageFilter , BinaryDilateImageFilter

     itk::simple::HConvex for the procedural interface

     itk::HConvexImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkHConvexImageFilter.h

    
### sitk.HMaxima
    HMaxima(Image image1, double height=2.0) -> Image



    Suppress local maxima whose height above the baseline is less than h.


    This function directly calls the execute method of HMaximaImageFilter in order to support a procedural API


    See:
     itk::simple::HMaximaImageFilter for the object oriented interface



    
### sitk.HMaximaImageFilter


    Suppress local maxima whose height above the baseline is less than h.


    HMaximaImageFilter suppresses local maxima that are less than h intensity units above
    the (local) background. This has the effect of smoothing over the
    "high" parts of the noise in the image without smoothing over large
    changes in intensity (region boundaries). See the HMinimaImageFilter to suppress the local minima whose depth is less than h intensity
    units below the (local) background.

    If the output of HMaximaImageFilter is subtracted from the original image, the signicant "peaks" in the
    image can be identified. This is what the HConvexImageFilter provides.

    This filter uses the ReconstructionByDilationImageFilter . It provides its own input as the "mask" input to the geodesic
    dilation. The "marker" image for the geodesic dilation is the input
    image minus the height parameter h.

    Geodesic morphology and the H-Maxima algorithm is described in Chapter
    6 of Pierre Soille's book "Morphological Image Analysis: Principles
    and Applications", Second Edition, Springer, 2003.

    The height parameter is set using SetHeight.


    See:
     ReconstructionByDilationImageFilter , HMinimaImageFilter , HConvexImageFilter

     MorphologyImageFilter , GrayscaleDilateImageFilter , GrayscaleFunctionDilateImageFilter , BinaryDilateImageFilter

     itk::simple::HMaxima for the procedural interface

     itk::HMaximaImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkHMaximaImageFilter.h

    
### sitk.HMinima
    HMinima(Image image1, double height=2.0, bool fullyConnected=False) -> Image



    Suppress local minima whose depth below the baseline is less than h.


    This function directly calls the execute method of HMinimaImageFilter in order to support a procedural API


    See:
     itk::simple::HMinimaImageFilter for the object oriented interface



    
### sitk.HMinimaImageFilter


    Suppress local minima whose depth below the baseline is less than h.


    HMinimaImageFilter suppresses local minima that are less than h intensity units below
    the (local) background. This has the effect of smoothing over the
    "low" parts of the noise in the image without smoothing over large
    changes in intensity (region boundaries). See the HMaximaImageFilter to suppress the local maxima whose height is less than h intensity
    units above the (local) background.

    If original image is subtracted from the output of HMinimaImageFilter , the signicant "valleys" in the image can be identified. This is
    what the HConcaveImageFilter provides.

    This filter uses the GrayscaleGeodesicErodeImageFilter . It provides its own input as the "mask" input to the geodesic
    dilation. The "marker" image for the geodesic dilation is the input
    image plus the height parameter h.

    Geodesic morphology and the H-Minima algorithm is described in Chapter
    6 of Pierre Soille's book "Morphological Image Analysis: Principles
    and Applications", Second Edition, Springer, 2003.


    See:
     GrayscaleGeodesicDilateImageFilter , HMinimaImageFilter , HConvexImageFilter

     MorphologyImageFilter , GrayscaleDilateImageFilter , GrayscaleFunctionDilateImageFilter , BinaryDilateImageFilter

     itk::simple::HMinima for the procedural interface

     itk::HMinimaImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkHMinimaImageFilter.h

    
### sitk.HalfHermitianToRealInverseFFT
    HalfHermitianToRealInverseFFT(Image image1, bool actualXDimensionIsOdd=False) -> Image



    Base class for specialized complex-to-real inverse Fast Fourier Transform .


    This function directly calls the execute method of HalfHermitianToRealInverseFFTImageFilter in order to support a procedural API


    See:
     itk::simple::HalfHermitianToRealInverseFFTImageFilter for the object oriented interface



    
### sitk.HalfHermitianToRealInverseFFTImageFilter


    Base class for specialized complex-to-real inverse Fast Fourier Transform .


    This is a base class for the "inverse" or "reverse" Discrete
    Fourier Transform . This is an abstract base class: the actual implementation is
    provided by the best child class available on the system when the
    object is created via the object factory system.

    The input to this filter is assumed to have the same format as the
    output of the RealToHalfHermitianForwardFFTImageFilter . That is, the input is assumed to consist of roughly half the full
    complex image resulting from a real-to-complex discrete Fourier
    transform. This half is expected to be the first half of the image in
    the X-dimension. Because this filter assumes that the input stores
    only about half of the non-redundant complex pixels, the output is
    larger in the X-dimension than it is in the input. To determine the
    actual size of the output image, this filter needs additional
    information in the form of a flag indicating whether the output image
    has an odd size in the X-dimension. Use SetActualXDimensionIsOdd() to set this flag.


    See:
     ForwardFFTImageFilter , HalfHermitianToRealInverseFFTImageFilter

     itk::simple::HalfHermitianToRealInverseFFT for the procedural interface

     itk::HalfHermitianToRealInverseFFTImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkHalfHermitianToRealInverseFFTImageFilter.h

    
### sitk.Hash
    Hash(Image image, itk::simple::HashImageFilter::HashFunction function) -> std::string



    
### sitk.HashImageFilter


    Compute the sha1 or md5 hash of an image.



    See:
     itk::simple::Hash for the procedural interface


    C++ includes: sitkHashImageFilter.h

    
### sitk.HausdorffDistanceImageFilter


    Computes the Hausdorff distance between the set of non-zero pixels of
    two images.


    HausdorffDistanceImageFilter computes the distance between the set non-zero pixels of two images
    using the following formula: \[ H(A,B) = \max(h(A,B),h(B,A)) \] where \[ h(A,B) = \max_{a \in A} \min_{b \in B} \| a -
    b\| \] is the directed Hausdorff distance and $A$ and $B$ are respectively the set of non-zero pixels in the first and second
    input images.

    In particular, this filter uses the DirectedHausdorffImageFilter
    inside to compute the two directed distances and then select the
    largest of the two.

    The Hausdorff distance measures the degree of mismatch between two
    sets and behaves like a metric over the set of all closed bounded sets
    - with properties of identity, symmetry and triangle inequality.

    This filter requires the largest possible region of the first image
    and the same corresponding region in the second image. It behaves as
    filter with two inputs and one output. Thus it can be inserted in a
    pipeline with other filters. The filter passes the first input through
    unmodified.

    This filter is templated over the two input image types. It assume
    both images have the same number of dimensions.


    See:
     DirectedHausdorffDistanceImageFilter

     itk::HausdorffDistanceImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkHausdorffDistanceImageFilter.h

    
### sitk.HistogramMatching
    HistogramMatching(Image image1, Image image2, uint32_t numberOfHistogramLevels=256, uint32_t numberOfMatchPoints=1, bool thresholdAtMeanIntensity=True) -> Image



    Normalize the grayscale values between two images by histogram
    matching.


    This function directly calls the execute method of HistogramMatchingImageFilter in order to support a procedural API


    See:
     itk::simple::HistogramMatchingImageFilter for the object oriented interface



    
### sitk.HistogramMatchingImageFilter


    Normalize the grayscale values between two images by histogram
    matching.


    HistogramMatchingImageFilter normalizes the grayscale values of a source image based on the
    grayscale values of a reference image. This filter uses a histogram
    matching technique where the histograms of the two images are matched
    only at a specified number of quantile values.

    This filter was originally designed to normalize MR images of the same
    MR protocol and same body part. The algorithm works best if background
    pixels are excluded from both the source and reference histograms. A
    simple background exclusion method is to exclude all pixels whose
    grayscale values are smaller than the mean grayscale value. ThresholdAtMeanIntensityOn() switches on this simple background exclusion method.

    The source image can be set via either SetInput() or SetSourceImage()
    . The reference image can be set via SetReferenceImage() .

    SetNumberOfHistogramLevels() sets the number of bins used when creating histograms of the source
    and reference images. SetNumberOfMatchPoints() governs the number of quantile values to be matched.

    This filter assumes that both the source and reference are of the same
    type and that the input and output image type have the same number of
    dimension and have scalar pixel types.

    REFERENCE
    Laszlo G. Nyul, Jayaram K. Udupa, and Xuan Zhang, "New Variants of a
    Method of MRI Scale Standardization", IEEE Transactions on Medical
    Imaging, 19(2):143-150, 2000.

    See:
     itk::simple::HistogramMatching for the procedural interface

     itk::HistogramMatchingImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkHistogramMatchingImageFilter.h

    
### sitk.HuangThreshold
    HuangThreshold(Image image, Image maskImage, uint8_t insideValue=1, uint8_t outsideValue=0, uint32_t numberOfHistogramBins=128, bool maskOutput=True, uint8_t maskValue=255) -> Image
    HuangThreshold(Image image, uint8_t insideValue=1, uint8_t outsideValue=0, uint32_t numberOfHistogramBins=128, bool maskOutput=True, uint8_t maskValue=255) -> Image



    
### sitk.HuangThresholdImageFilter


    Threshold an image using the Huang Threshold.


    This filter creates a binary thresholded image that separates an image
    into foreground and background components. The filter computes the
    threshold using the HuangThresholdCalculator and applies that threshold to the input image using the BinaryThresholdImageFilter .


    Richard Beare. Department of Medicine, Monash University, Melbourne,
    Australia.

    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.
     This implementation was taken from the Insight Journal paper: https://hdl.handle.net/10380/3279 or http://www.insight-journal.org/browse/publication/811


    See:
     HistogramThresholdImageFilter

     itk::simple::HuangThreshold for the procedural interface

     itk::HuangThresholdImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkHuangThresholdImageFilter.h

    
### sitk.Image


    The Image class for SimpleITK.


    This Image class can represent 2D, 3D, and 4D images. The pixel types may be a
    scalar, a multi-component vector or a run-length-encoded (RLE)
    "label". The dimension, pixel type and size is specified at
    construction.

    A fundamental concept of ITK images is that they occupy physical space
    where the image is defined by an origin, spacing, and direction cosine
    matrix. The attributes are taken into consideration when doing most
    operations on an image. A meta-data dictionary is also associated with
    the image, which may contain additional fields from reading but these
    attributes are not propagated by image filters.

    The SimpleITK Image provides a single facade interface to several ITK image types.
    Internally, the SimpleITK Image maintains a pointer to the ITK image class, and performs reference
    counting and lazy copying. This means that deep copying of an image
    including it's buffer is delayed until the image is modified. This
    removes the need to use pointers to SimpleITK Image class, as copying and returning by value do not unnecessarily
    duplicate the data.

    /sa itk::Image itk::VectorImage itk::LabelMap itk::ImageBase

    C++ includes: sitkImage.h

    
### sitk.ImageFileReader


    Read an image file and return a SimpleITK Image.


    The reader can handle scalar images, and vector images. Pixel types
    such as RGB, RGBA are loaded as multi-component images with vector
    pixel types. Additionally, tensor images are loaded with the pixel
    type being a 1-d vector.

    An interface is also provided to access the information from the
    underlying itk::ImageIO. This information can be loaded with the
    ReadImageInformation method.

    Reading takes place by the ITK ImageIO factory mechanism. ITK contains
    many ImageIO classes which are responsible for reading separate file
    formats. By default, each ImageIO is asked if it "can read" the
    file, and the first one which "can read" the format is used. The
    list of available ImageIOs can be obtained using the
    GetRegisteredImageIOs method. The ImageIO used can be overridden with
    the SetImageIO method. This is useful in cases when multiple ImageIOs
    "can read" the file and the user wants to select a specific IO (not
    the first).


    See:
     itk::simple::ReadImage for the procedural interface


    C++ includes: sitkImageFileReader.h

    
### sitk.ImageFileWriter


    Write out a SimpleITK image to the specified file location.


    This writer tries to write the image out using the image's type to the
    location specified in FileName. If writing fails, an ITK exception is
    thrown.


    See:
     itk::simple::WriteImage for the procedural interface


    C++ includes: sitkImageFileWriter.h

    
### sitk.ImageFilter_0


    The base interface for SimpleITK filters that take one input image.


    All SimpleITK filters which take one input image should inherit from
    this class

    C++ includes: sitkImageFilter.h

    
### sitk.ImageFilter_1


    The base interface for SimpleITK filters that take one input image.


    All SimpleITK filters which take one input image should inherit from
    this class

    C++ includes: sitkImageFilter.h

    
### sitk.ImageFilter_2


    The base interface for SimpleITK filters that take one input image.


    All SimpleITK filters which take one input image should inherit from
    this class

    C++ includes: sitkImageFilter.h

    
### sitk.ImageFilter_3


    The base interface for SimpleITK filters that take one input image.


    All SimpleITK filters which take one input image should inherit from
    this class

    C++ includes: sitkImageFilter.h

    
### sitk.ImageFilter_4


    The base interface for SimpleITK filters that take one input image.


    All SimpleITK filters which take one input image should inherit from
    this class

    C++ includes: sitkImageFilter.h

    
### sitk.ImageFilter_5


    The base interface for SimpleITK filters that take one input image.


    All SimpleITK filters which take one input image should inherit from
    this class

    C++ includes: sitkImageFilter.h

    
### sitk.ImageReaderBase


    An abract base class for image readers.

    C++ includes: sitkImageReaderBase.h

    
### sitk.ImageRegistrationMethod


    An interface method to the modular ITKv4 registration framework.


    This interface method class encapsulates typical registration usage by
    incorporating all the necessary elements for performing a simple image
    registration between two images. This method also allows for
    multistage registration whereby each stage is characterized by
    possibly different transforms and different image metrics. For
    example, many users will want to perform a linear registration
    followed by deformable registration where both stages are performed in
    multiple levels. Each level can be characterized by:


    the resolution of the virtual domain image (see below)

    smoothing of the fixed and moving images
     Multiple stages are handled by linking multiple instantiations of
    this class where the output transform is added to the optional
    composite transform input.


    See:
     itk::ImageRegistrationMethodv4

     itk::ImageToImageMetricv4

     itk::ObjectToObjectOptimizerBaseTemplate


    C++ includes: sitkImageRegistrationMethod.h

    
### sitk.ImageSeriesReader


    Read series of image files into a SimpleITK image.


    For some image formats such as DICOM, images also contain associated
    meta-data (e.g. imaging modality, patient name etc.). By default the
    reader does not load this information (saves time). To load the meta-
    data you will need to explicitly configure the reader,
    MetaDataDictionaryArrayUpdateOn, and possibly specify that you also
    want to load the private meta-data LoadPrivateTagsOn.

    Once the image series is read the meta-data is directly accessible
    from the reader.


    See:
     itk::simple::ReadImage for the procedural interface


    C++ includes: sitkImageSeriesReader.h

    
### sitk.ImageSeriesReader_GetGDCMSeriesFileNamesImageSeriesReader_GetGDCMSeriesFileNames(std::string const & directory, std::string const & seriesID, bool useSeriesDetails=False, bool recursive=False, bool loadSequences=False) -> VectorString
### sitk.ImageSeriesReader_GetGDCMSeriesIDsImageSeriesReader_GetGDCMSeriesIDs(std::string const & directory) -> VectorString
### sitk.ImageSeriesWriter


    Writer series of image from a SimpleITK image.


    The ImageSeriesWriter is for writing a 3D image as a series of 2D images. A list of names
    for the series of 2D images must be provided, and an exception will be
    generated if the number of file names does not match the size of the
    image in the z-direction.

    DICOM series cannot be written with this class, as an exception will
    be generated. To write a DICOM series the individual slices must be
    extracted, proper DICOM tags must be added to the dictionaries, then
    written with the ImageFileWriter.


    See:
     itk::simple::WriteImage for the procedural interface


    C++ includes: sitkImageSeriesWriter.h

    
### sitk.ImageViewer


    Display an image in an external viewer (Fiji by default)


    The ImageViewer class displays an image with an external image display application.
    By default the class will search for a Fiji ( https://fiji.sc ) executable. The image is written out to a temporary file and then
    passed to the application.

    When SimpleITK is first invoked the following environment variables
    are queried to set up the external viewer:

    SITK_SHOW_EXTENSION: file format extension of the temporary image
    file. The default is '.mha', the MetaIO file format.

    SITK_SHOW_COMMAND: The user can specify an application other than Fiji
    to view images.

    These environment variables are only checked at SimpleITK's launch.

    C++ includes: sitkImageViewer.h

    
### sitk.ImageViewer_GetGlobalDefaultApplicationImageViewer_GetGlobalDefaultApplication() -> std::string const &
### sitk.ImageViewer_GetGlobalDefaultDebugImageViewer_GetGlobalDefaultDebug() -> bool
### sitk.ImageViewer_GetGlobalDefaultExecutableNamesImageViewer_GetGlobalDefaultExecutableNames() -> VectorString
### sitk.ImageViewer_GetGlobalDefaultFileExtensionImageViewer_GetGlobalDefaultFileExtension() -> std::string const &
### sitk.ImageViewer_GetGlobalDefaultSearchPathImageViewer_GetGlobalDefaultSearchPath() -> VectorString
### sitk.ImageViewer_GetProcessDelayImageViewer_GetProcessDelay() -> unsigned int
### sitk.ImageViewer_SetGlobalDefaultApplicationImageViewer_SetGlobalDefaultApplication(std::string const & app)
### sitk.ImageViewer_SetGlobalDefaultDebugImageViewer_SetGlobalDefaultDebug(bool const dbg)
### sitk.ImageViewer_SetGlobalDefaultDebugOffImageViewer_SetGlobalDefaultDebugOff()
### sitk.ImageViewer_SetGlobalDefaultDebugOnImageViewer_SetGlobalDefaultDebugOn()
### sitk.ImageViewer_SetGlobalDefaultExecutableNamesImageViewer_SetGlobalDefaultExecutableNames(VectorString names)
### sitk.ImageViewer_SetGlobalDefaultFileExtensionImageViewer_SetGlobalDefaultFileExtension(std::string const & ext)
### sitk.ImageViewer_SetGlobalDefaultSearchPathImageViewer_SetGlobalDefaultSearchPath(VectorString path)
### sitk.ImageViewer_SetProcessDelayImageViewer_SetProcessDelay(unsigned int const delay)
### sitk.IntensityWindowing
    IntensityWindowing(Image image1, double windowMinimum=0.0, double windowMaximum=255, double outputMinimum=0, double outputMaximum=255) -> Image



    Applies a linear transformation to the intensity levels of the input Image that are inside a user-defined interval. Values below this interval
    are mapped to a constant. Values over the interval are mapped to
    another constant.


    This function directly calls the execute method of IntensityWindowingImageFilter in order to support a procedural API


    See:
     itk::simple::IntensityWindowingImageFilter for the object oriented interface



    
### sitk.IntensityWindowingImageFilter


    Applies a linear transformation to the intensity levels of the input Image that are inside a user-defined interval. Values below this interval
    are mapped to a constant. Values over the interval are mapped to
    another constant.


    IntensityWindowingImageFilter applies pixel-wise a linear transformation to the intensity values of
    input image pixels. The linear transformation is defined by the user
    in terms of the minimum and maximum values that the output image
    should have and the lower and upper limits of the intensity window of
    the input image. This operation is very common in visualization, and
    can also be applied as a convenient preprocessing operation for image
    segmentation.

    All computations are performed in the precision of the input pixel's
    RealType. Before assigning the computed value to the output pixel.


    See:
     RescaleIntensityImageFilter

     itk::simple::IntensityWindowing for the procedural interface

     itk::IntensityWindowingImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkIntensityWindowingImageFilter.h

    
### sitk.IntermodesThreshold
    IntermodesThreshold(Image image, Image maskImage, uint8_t insideValue=1, uint8_t outsideValue=0, uint32_t numberOfHistogramBins=256, bool maskOutput=True, uint8_t maskValue=255) -> Image
    IntermodesThreshold(Image image, uint8_t insideValue=1, uint8_t outsideValue=0, uint32_t numberOfHistogramBins=256, bool maskOutput=True, uint8_t maskValue=255) -> Image



    
### sitk.IntermodesThresholdImageFilter


    Threshold an image using the Intermodes Threshold.


    This filter creates a binary thresholded image that separates an image
    into foreground and background components. The filter computes the
    threshold using the IntermodesThresholdCalculator and applies that threshold to the input image using the BinaryThresholdImageFilter .


    Richard Beare. Department of Medicine, Monash University, Melbourne,
    Australia.

    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.
     This implementation was taken from the Insight Journal paper: https://hdl.handle.net/10380/3279 or http://www.insight-journal.org/browse/publication/811


    See:
     HistogramThresholdImageFilter

     itk::simple::IntermodesThreshold for the procedural interface

     itk::IntermodesThresholdImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkIntermodesThresholdImageFilter.h

    
### sitk.InverseDeconvolution
    InverseDeconvolution(Image image1, Image image2, double kernelZeroMagnitudeThreshold=1.0e-4, bool normalize=False, itk::simple::InverseDeconvolutionImageFilter::BoundaryConditionType boundaryCondition, itk::simple::InverseDeconvolutionImageFilter::OutputRegionModeType outputRegionMode) -> Image



    The direct linear inverse deconvolution filter.


    This function directly calls the execute method of InverseDeconvolutionImageFilter in order to support a procedural API


    See:
     itk::simple::InverseDeconvolutionImageFilter for the object oriented interface



    
### sitk.InverseDeconvolutionImageFilter


    The direct linear inverse deconvolution filter.


    The inverse filter is the most straightforward deconvolution method.
    Considering that convolution of two images in the spatial domain is
    equivalent to multiplying the Fourier transform of the two images, the
    inverse filter consists of inverting the multiplication. In other
    words, this filter computes the following: \[ hat{F}(\omega) = \begin{cases} G(\omega) / H(\omega)
    & \text{if \f$|H(\omega)| \geq \epsilon\f$} \\
    0 & \text{otherwise} \end{cases} \] where $\hat{F}(\omega)$ is the Fourier transform of the estimate produced by this filter, $G(\omega)$ is the Fourier transform of the input blurred image, $H(\omega)$ is the Fourier transform of the blurring kernel, and $\epsilon$ is a constant real non-negative threshold (called
    KernelZeroMagnitudeThreshold in this filter) that determines when the
    magnitude of a complex number is considered zero.


    Gaetan Lehmann, Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France

    Cory Quammen, The University of North Carolina at Chapel Hill

    See:
     itk::simple::InverseDeconvolution for the procedural interface

     itk::InverseDeconvolutionImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkInverseDeconvolutionImageFilter.h

    
### sitk.InverseDisplacementField
    InverseDisplacementField(Image image1, VectorUInt32 size, VectorDouble outputOrigin, VectorDouble outputSpacing, unsigned int subsamplingFactor=16) -> Image



    Computes the inverse of a displacement field.


    This function directly calls the execute method of InverseDisplacementFieldImageFilter in order to support a procedural API


    See:
     itk::simple::InverseDisplacementFieldImageFilter for the object oriented interface



    
### sitk.InverseDisplacementFieldImageFilter


    Computes the inverse of a displacement field.


    InverseDisplacementFieldImageFilter takes a displacement field as input and computes the displacement
    field that is its inverse. If the input displacement field was mapping
    coordinates from a space A into a space B, the output of this filter
    will map coordinates from the space B into the space A.

    Given that both the input and output displacement field are
    represented as discrete images with pixel type vector, the inverse
    will be only an estimation and will probably not correspond to a
    perfect inverse. The precision of the inverse can be improved at the
    price of increasing the computation time and memory consumption in
    this filter.

    The method used for computing the inverse displacement field is to
    subsample the input field using a regular grid and create Kerned-Base
    Spline in which the reference landmarks are the coordinates of the
    deformed point and the target landmarks are the negative of the
    displacement vectors. The kernel-base spline is then used for
    regularly sampling the output space and recover vector values for
    every single pixel.

    The subsampling factor used for the regular grid of the input field
    will determine the number of landmarks in the KernelBased spline and
    therefore it will have a dramatic effect on both the precision of
    output displacement field and the computational time required for the
    filter to complete the estimation. A large subsampling factor will
    result in few landmarks in the KernelBased spline, therefore on fast
    computation and low precision. A small subsampling factor will result
    in a large number of landmarks in the KernelBased spline, therefore a
    large memory consumption, long computation time and high precision for
    the inverse estimation.

    This filter expects both the input and output images to be of pixel
    type Vector .
    See:
     itk::simple::InverseDisplacementField for the procedural interface

     itk::InverseDisplacementFieldImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkInverseDisplacementFieldImageFilter.h

    
### sitk.InverseFFT
    InverseFFT(Image image1) -> Image



    Base class for inverse Fast Fourier Transform .


    This function directly calls the execute method of InverseFFTImageFilter in order to support a procedural API


    See:
     itk::simple::InverseFFTImageFilter for the object oriented interface



    
### sitk.InverseFFTImageFilter


    Base class for inverse Fast Fourier Transform .


    This is a base class for the "inverse" or "reverse" Discrete
    Fourier Transform . This is an abstract base class: the actual implementation is
    provided by the best child available on the system when the object is
    created via the object factory system.

    This class transforms a full complex image with Hermitian symmetry
    into its real spatial domain representation. If the input does not
    have Hermitian symmetry, the imaginary component is discarded.


    See:
     ForwardFFTImageFilter , InverseFFTImageFilter

     itk::simple::InverseFFT for the procedural interface

     itk::InverseFFTImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkInverseFFTImageFilter.h

    
### sitk.InvertDisplacementField
    InvertDisplacementField(Image image1, uint32_t maximumNumberOfIterations=10, double maxErrorToleranceThreshold=0.1, double meanErrorToleranceThreshold=0.001, bool enforceBoundaryCondition=True) -> Image



    Iteratively estimate the inverse field of a displacement field.


    This function directly calls the execute method of InvertDisplacementFieldImageFilter in order to support a procedural API


    See:
     itk::simple::InvertDisplacementFieldImageFilter for the object oriented interface



    
### sitk.InvertDisplacementFieldImageFilter


    Iteratively estimate the inverse field of a displacement field.



    Nick Tustison

    Brian Avants

    See:
     itk::simple::InvertDisplacementField for the procedural interface

     itk::InvertDisplacementFieldImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkInvertDisplacementFieldImageFilter.h

    
### sitk.InvertIntensity
    InvertIntensity(Image image1, double maximum=255) -> Image



    Invert the intensity of an image.


    This function directly calls the execute method of InvertIntensityImageFilter in order to support a procedural API


    See:
     itk::simple::InvertIntensityImageFilter for the object oriented interface



    
### sitk.InvertIntensityImageFilter


    Invert the intensity of an image.


    InvertIntensityImageFilter inverts intensity of pixels by subtracting pixel value to a maximum
    value. The maximum value can be set with SetMaximum and defaults the
    maximum of input pixel type. This filter can be used to invert, for
    example, a binary image, a distance map, etc.


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.

    See:
     IntensityWindowingImageFilter ShiftScaleImageFilter

     itk::simple::InvertIntensity for the procedural interface

     itk::InvertIntensityImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkInvertIntensityImageFilter.h

    
### sitk.IsoContourDistance
    IsoContourDistance(Image image1, double levelSetValue=0.0, double farValue=10) -> Image



    Compute an approximate distance from an interpolated isocontour to the
    close grid points.


    This function directly calls the execute method of IsoContourDistanceImageFilter in order to support a procedural API


    See:
     itk::simple::IsoContourDistanceImageFilter for the object oriented interface



    
### sitk.IsoContourDistanceImageFilter


    Compute an approximate distance from an interpolated isocontour to the
    close grid points.


    For standard level set algorithms, it is useful to periodically
    reinitialize the evolving image to prevent numerical accuracy problems
    in computing derivatives. This reinitialization is done by computing a
    signed distance map to the current level set. This class provides the
    first step in this reinitialization by computing an estimate of the
    distance from the interpolated isocontour to the pixels (or voxels)
    that are close to it, i.e. for which the isocontour crosses a segment
    between them and one of their direct neighbors. This class supports
    narrowbanding. If the input narrowband is provided, the algorithm will
    only locate the level set within the input narrowband.

    Implementation of this class is based on Fast and Accurate
    Redistancing for Level Set Methods `Krissian K. and Westin C.F.',
    EUROCAST NeuroImaging Workshop Las Palmas Spain, Ninth International
    Conference on Computer Aided Systems Theory , pages 48-51, Feb 2003.
    See:
     itk::simple::IsoContourDistance for the procedural interface

     itk::IsoContourDistanceImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkIsoContourDistanceImageFilter.h

    
### sitk.IsoDataThreshold
    IsoDataThreshold(Image image, Image maskImage, uint8_t insideValue=1, uint8_t outsideValue=0, uint32_t numberOfHistogramBins=256, bool maskOutput=True, uint8_t maskValue=255) -> Image
    IsoDataThreshold(Image image, uint8_t insideValue=1, uint8_t outsideValue=0, uint32_t numberOfHistogramBins=256, bool maskOutput=True, uint8_t maskValue=255) -> Image



    
### sitk.IsoDataThresholdImageFilter


    Threshold an image using the IsoData Threshold.


    This filter creates a binary thresholded image that separates an image
    into foreground and background components. The filter computes the
    threshold using the IsoDataThresholdCalculator and applies that threshold to the input image using the BinaryThresholdImageFilter .


    Richard Beare. Department of Medicine, Monash University, Melbourne,
    Australia.

    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.
     This implementation was taken from the Insight Journal paper: https://hdl.handle.net/10380/3279 or http://www.insight-journal.org/browse/publication/811


    See:
     HistogramThresholdImageFilter

     itk::simple::IsoDataThreshold for the procedural interface

     itk::IsoDataThresholdImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkIsoDataThresholdImageFilter.h

    
### sitk.IsolatedConnected
    IsolatedConnected(Image image1, VectorUInt32 seed1, VectorUInt32 seed2, double lower=0, double upper=1, uint8_t replaceValue=1, double isolatedValueTolerance=1.0, bool findUpperThreshold=True) -> Image



    Label pixels that are connected to one set of seeds but not another.


    This function directly calls the execute method of IsolatedConnectedImageFilter in order to support a procedural API


    See:
     itk::simple::IsolatedConnectedImageFilter for the object oriented interface



    
### sitk.IsolatedConnectedImageFilter


    Label pixels that are connected to one set of seeds but not another.


    IsolatedConnectedImageFilter finds the optimal threshold to separate two regions. It has two
    modes, one to separate dark regions surrounded by bright regions by
    automatically finding a minimum isolating upper threshold, and another
    to separate bright regions surrounded by dark regions by automatically
    finding a maximum lower isolating threshold. The mode can be chosen by
    setting FindUpperThresholdOn() /Off(). In both cases, the isolating threshold is retrieved with GetIsolatedValue() .

    The algorithm labels pixels with ReplaceValue that are connected to
    Seeds1 AND NOT connected to Seeds2. When finding the threshold to
    separate two dark regions surrounded by bright regions, given a fixed
    lower threshold, the filter adjusts the upper threshold until the two
    sets of seeds are not connected. The algorithm uses a binary search to
    adjust the upper threshold, starting at Upper. The reverse is true for
    finding the threshold to separate two bright regions. Lower defaults
    to the smallest possible value for the InputImagePixelType, and Upper
    defaults to the largest possible value for the InputImagePixelType.

    The user can also supply the Lower and Upper values to restrict the
    search. However, if the range is too restrictive, it could happen that
    no isolating threshold can be found between the user specified Lower
    and Upper values. Therefore, unless the user is sure of the bounds to
    set, it is recommended that the user set these values to the lowest
    and highest intensity values in the image, respectively.

    The user can specify more than one seed for both regions to separate.
    The algorithm will try find the threshold that ensures that all of the
    first seeds are contained in the resulting segmentation and all of the
    second seeds are not contained in the segmentation.

    It is possible that the algorithm may not be able to find the
    isolating threshold because no such threshold exists. The user can
    check for this by querying the GetThresholdingFailed() flag.
    See:
     itk::simple::IsolatedConnected for the procedural interface

     itk::IsolatedConnectedImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkIsolatedConnectedImageFilter.h

    
### sitk.IsolatedWatershed
    IsolatedWatershed(Image image1, VectorUInt32 seed1, VectorUInt32 seed2, double threshold=0.0, double upperValueLimit=1.0, double isolatedValueTolerance=0.001, uint8_t replaceValue1=1, uint8_t replaceValue2=2) -> Image



    Isolate watershed basins using two seeds.


    This function directly calls the execute method of IsolatedWatershedImageFilter in order to support a procedural API


    See:
     itk::simple::IsolatedWatershedImageFilter for the object oriented interface



    
### sitk.IsolatedWatershedImageFilter


    Isolate watershed basins using two seeds.


    IsolatedWatershedImageFilter labels pixels with ReplaceValue1 that are in the same watershed basin
    as Seed1 AND NOT the same as Seed2. The filter adjusts the waterlevel
    until the two seeds are not in different basins. The user supplies a
    Watershed threshold. The algorithm uses a binary search to adjust the
    upper waterlevel, starting at UpperValueLimit. UpperValueLimit
    defaults to the 1.0.
    See:
     itk::simple::IsolatedWatershed for the procedural interface

     itk::IsolatedWatershedImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkIsolatedWatershedImageFilter.h

    
### sitk.IterativeInverseDisplacementField
    IterativeInverseDisplacementField(Image image1, uint32_t numberOfIterations=5, double stopValue=0.0) -> Image



    Computes the inverse of a displacement field.


    This function directly calls the execute method of IterativeInverseDisplacementFieldImageFilter in order to support a procedural API


    See:
     itk::simple::IterativeInverseDisplacementFieldImageFilter for the object oriented interface



    
### sitk.IterativeInverseDisplacementFieldImageFilter


    Computes the inverse of a displacement field.


    IterativeInverseDisplacementFieldImageFilter takes a displacement field as input and computes the displacement
    field that is its inverse. If the input displacement field was mapping
    coordinates from a space A into a space B, the output of this filter
    will map coordinates from the space B into the space A.

    The algorithm implemented in this filter uses an iterative method for
    progresively refining the values of the inverse field. Starting from
    the direct field, at every pixel the direct mapping of this point is
    found, and a the nevative of the current displacement is stored in the
    inverse field at the nearest pixel. Then, subsequent iterations verify
    if any of the neigbor pixels provide a better return to the current
    pixel, in which case its value is taken for updating the vector in the
    inverse field.

    This method was discussed in the users-list during February 2004.


    Corinne Mattmann

    See:
     itk::simple::IterativeInverseDisplacementField for the procedural interface

     itk::IterativeInverseDisplacementFieldImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkIterativeInverseDisplacementFieldImageFilter.h

    
### sitk.JoinSeries
    JoinSeries(VectorOfImage images, double origin=0.0, double spacing=1.0) -> Image
    JoinSeries(Image image1, double origin=0.0, double spacing=1.0) -> Image
    JoinSeries(Image image1, Image image2, double origin=0.0, double spacing=1.0) -> Image
    JoinSeries(Image image1, Image image2, Image image3, double origin=0.0, double spacing=1.0) -> Image
    JoinSeries(Image image1, Image image2, Image image3, Image image4, double origin=0.0, double spacing=1.0) -> Image
    JoinSeries(Image image1, Image image2, Image image3, Image image4, Image image5, double origin=0.0, double spacing=1.0) -> Image
    
### sitk.JoinSeriesImageFilter


    Join N-D images into an (N+1)-D image.


    This filter is templated over the input image type and the output
    image type. The pixel type of them must be the same and the input
    dimension must be less than the output dimension. When the input
    images are N-dimensinal, they are joined in order and the size of the
    N+1'th dimension of the output is same as the number of the inputs.
    The spacing and the origin (where the first input is placed) for the
    N+1'th dimension is specified in this filter. The output image
    informations for the first N dimensions are taken from the first
    input. Note that all the inputs should have the same information.


    Hideaki Hiraki
     Contributed in the users list http://public.kitware.com/pipermail/insight-
    users/2004-February/006542.html


    See:
     itk::simple::JoinSeries for the procedural interface


    C++ includes: sitkJoinSeriesImageFilter.h

    
### sitk.KittlerIllingworthThreshold
    KittlerIllingworthThreshold(Image image, Image maskImage, uint8_t insideValue=1, uint8_t outsideValue=0, uint32_t numberOfHistogramBins=256, bool maskOutput=True, uint8_t maskValue=255) -> Image
    KittlerIllingworthThreshold(Image image, uint8_t insideValue=1, uint8_t outsideValue=0, uint32_t numberOfHistogramBins=256, bool maskOutput=True, uint8_t maskValue=255) -> Image



    
### sitk.KittlerIllingworthThresholdImageFilter


    Threshold an image using the KittlerIllingworth Threshold.


    This filter creates a binary thresholded image that separates an image
    into foreground and background components. The filter computes the
    threshold using the KittlerIllingworthThresholdCalculator and applies that threshold to the input image using the BinaryThresholdImageFilter .


    Richard Beare. Department of Medicine, Monash University, Melbourne,
    Australia.

    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.
     This implementation was taken from the Insight Journal paper: https://hdl.handle.net/10380/3279 or http://www.insight-journal.org/browse/publication/811


    See:
     HistogramThresholdImageFilter

     itk::simple::KittlerIllingworthThreshold for the procedural interface

     itk::KittlerIllingworthThresholdImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkKittlerIllingworthThresholdImageFilter.h

    
### sitk.LabelContour
    LabelContour(Image image1, bool fullyConnected=False, double backgroundValue=0) -> Image



    Labels the pixels on the border of the objects in a labeled image.


    This function directly calls the execute method of LabelContourImageFilter in order to support a procedural API


    See:
     itk::simple::LabelContourImageFilter for the object oriented interface



    
### sitk.LabelContourImageFilter


    Labels the pixels on the border of the objects in a labeled image.


    LabelContourImageFilter takes a labeled image as input, where the pixels in the objects are
    the pixels with a value different of the BackgroundValue. Only the
    pixels on the contours of the objects are kept. The pixels not on the
    border are changed to BackgroundValue. The labels of the object are
    the same in the input and in the output image.

    The connectivity can be changed to minimum or maximum connectivity
    with SetFullyConnected() . Full connectivity produces thicker contours.

    https://hdl.handle.net/1926/1352


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.

    See:
     BinaryContourImageFilter

     itk::simple::LabelContour for the procedural interface

     itk::LabelContourImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkLabelContourImageFilter.h

    
### sitk.LabelImageToLabelMap
    LabelImageToLabelMap(Image image1, double backgroundValue=0) -> Image



    convert a labeled image to a label collection image


    This function directly calls the execute method of LabelImageToLabelMapFilter in order to support a procedural API


    See:
     itk::simple::LabelImageToLabelMapFilter for the object oriented interface



    
### sitk.LabelImageToLabelMapFilter


    convert a labeled image to a label collection image


    LabelImageToLabelMapFilter converts a label image to a label collection image. The labels are
    the same in the input and the output image.


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.
     This implementation was taken from the Insight Journal paper: https://hdl.handle.net/1926/584 or http://www.insight-journal.org/browse/publication/176


    See:
     BinaryImageToLabelMapFilter , LabelMapToLabelImageFilter

     itk::simple::LabelImageToLabelMapFilter for the procedural interface

     itk::LabelImageToLabelMapFilter for the Doxygen on the original ITK class.


    C++ includes: sitkLabelImageToLabelMapFilter.h

    
### sitk.LabelIntensityStatisticsImageFilter


    a convenient class to convert a label image to a label map and valuate
    the statistics attributes at once



    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.
     This implementation was taken from the Insight Journal paper: https://hdl.handle.net/1926/584 or http://www.insight-journal.org/browse/publication/176


    See:
     StatisticsLabelObject , LabelStatisticsOpeningImageFilter , LabelStatisticsOpeningImageFilter

     itk::LabelImageToStatisticsLabelMapFilter for the Doxygen on the original ITK class.


    C++ includes: sitkLabelIntensityStatisticsImageFilter.h

    
### sitk.LabelMapContourOverlay
    LabelMapContourOverlay(Image labelMapImage, Image featureImage, double opacity=0.5, VectorUInt32 dilationRadius, VectorUInt32 contourThickness, unsigned int sliceDimension=0, itk::simple::LabelMapContourOverlayImageFilter::ContourTypeType contourType, itk::simple::LabelMapContourOverlayImageFilter::PriorityType priority, VectorUInt8 colormap) -> Image



    Apply a colormap to the contours (outlines) of each object in a label
    map and superimpose it on top of the feature image.


    This function directly calls the execute method of LabelMapContourOverlayImageFilter in order to support a procedural API


    See:
     itk::simple::LabelMapContourOverlayImageFilter for the object oriented interface



    
### sitk.LabelMapContourOverlayImageFilter


    Apply a colormap to the contours (outlines) of each object in a label
    map and superimpose it on top of the feature image.


    The feature image is typically the image from which the labeling was
    produced. Use the SetInput function to set the LabelMap , and the SetFeatureImage function to set the feature image.

    Apply a colormap to a label map and put it on top of the input image.
    The set of colors is a good selection of distinct colors. The opacity
    of the label map can be defined by the user. A background label
    produce a gray pixel with the same intensity than the input one.


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.
     This implementation was taken from the Insight Journal paper: https://hdl.handle.net/1926/584 or http://www.insight-journal.org/browse/publication/176


    See:
     LabelMapOverlayImageFilter , LabelOverlayImageFilter , LabelOverlayFunctor

     LabelMapToBinaryImageFilter , LabelMapToLabelImageFilter ,

     itk::simple::LabelMapContourOverlay for the procedural interface

     itk::LabelMapContourOverlayImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkLabelMapContourOverlayImageFilter.h

    
### sitk.LabelMapMask
    LabelMapMask(Image labelMapImage, Image featureImage, uint64_t label=1, double backgroundValue=0, bool negated=False, bool crop=False, VectorUInt32 cropBorder) -> Image



    Mask and image with a LabelMap .


    This function directly calls the execute method of LabelMapMaskImageFilter in order to support a procedural API


    See:
     itk::simple::LabelMapMaskImageFilter for the object oriented interface



    
### sitk.LabelMapMaskImageFilter


    Mask and image with a LabelMap .


    LabelMapMaskImageFilter mask the content of an input image according to the content of the
    input LabelMap . The masked pixel of the input image are set to the BackgroundValue. LabelMapMaskImageFilter can keep the input image for one label only, with Negated = false
    (the default) or it can mask the input image for a single label, when
    Negated equals true. In Both cases, the label is set with SetLabel() .


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.
     This implementation was taken from the Insight Journal paper: https://hdl.handle.net/1926/584 or http://www.insight-journal.org/browse/publication/176


    See:
     LabelMapToBinaryImageFilter , LabelMapToLabelImageFilter

     itk::simple::LabelMapMask for the procedural interface

     itk::LabelMapMaskImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkLabelMapMaskImageFilter.h

    
### sitk.LabelMapOverlay
    LabelMapOverlay(Image labelMapImage, Image featureImage, double opacity=0.5, VectorUInt8 colormap) -> Image



    Apply a colormap to a label map and superimpose it on an image.


    This function directly calls the execute method of LabelMapOverlayImageFilter in order to support a procedural API


    See:
     itk::simple::LabelMapOverlayImageFilter for the object oriented interface



    
### sitk.LabelMapOverlayImageFilter


    Apply a colormap to a label map and superimpose it on an image.


    Apply a colormap to a label map and put it on top of the feature
    image. The feature image is typically the image from which the
    labeling was produced. Use the SetInput function to set the LabelMap , and the SetFeatureImage function to set the feature image.

    The set of colors is a good selection of distinct colors. The opacity
    of the label map can be defined by the user. A background label
    produce a gray pixel with the same intensity than the input one.


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.
     This implementation was taken from the Insight Journal paper: https://hdl.handle.net/1926/584 or http://www.insight-journal.org/browse/publication/176


    See:
     LabelOverlayImageFilter , LabelOverlayFunctor

     LabelMapToRGBImageFilter , LabelMapToBinaryImageFilter , LabelMapToLabelImageFilter

     itk::simple::LabelMapOverlay for the procedural interface

     itk::LabelMapOverlayImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkLabelMapOverlayImageFilter.h

    
### sitk.LabelMapToBinary
    LabelMapToBinary(Image image1, double backgroundValue=0, double foregroundValue=1.0) -> Image



    Convert a LabelMap to a binary image.


    This function directly calls the execute method of LabelMapToBinaryImageFilter in order to support a procedural API


    See:
     itk::simple::LabelMapToBinaryImageFilter for the object oriented interface



    
### sitk.LabelMapToBinaryImageFilter


    Convert a LabelMap to a binary image.


    LabelMapToBinaryImageFilter to a binary image. All the objects in the image are used as
    foreground. The background values of the original binary image can be
    restored by passing this image to the filter with the
    SetBackgroundImage() method.

    This implementation was taken from the Insight Journal paper: https://hdl.handle.net/1926/584 or http://www.insight-journal.org/browse/publication/176


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.

    See:
     LabelMapToLabelImageFilter , LabelMapMaskImageFilter

     itk::simple::LabelMapToBinary for the procedural interface

     itk::LabelMapToBinaryImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkLabelMapToBinaryImageFilter.h

    
### sitk.LabelMapToLabel
    LabelMapToLabel(Image image1) -> Image



    Converts a LabelMap to a labeled image.


    This function directly calls the execute method of LabelMapToLabelImageFilter in order to support a procedural API


    See:
     itk::simple::LabelMapToLabelImageFilter for the object oriented interface



    
### sitk.LabelMapToLabelImageFilter


    Converts a LabelMap to a labeled image.


    LabelMapToBinaryImageFilter to a label image.


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.
     This implementation was taken from the Insight Journal paper: https://hdl.handle.net/1926/584 or http://www.insight-journal.org/browse/publication/176


    See:
     LabelMapToBinaryImageFilter , LabelMapMaskImageFilter

     itk::simple::LabelMapToLabel for the procedural interface

     itk::LabelMapToLabelImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkLabelMapToLabelImageFilter.h

    
### sitk.LabelMapToRGB
    LabelMapToRGB(Image image1, VectorUInt8 colormap) -> Image



    Convert a LabelMap to a colored image.


    This function directly calls the execute method of LabelMapToRGBImageFilter in order to support a procedural API


    See:
     itk::simple::LabelMapToRGBImageFilter for the object oriented interface



    
### sitk.LabelMapToRGBImageFilter


    Convert a LabelMap to a colored image.



    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.
     This implementation was taken from the Insight Journal paper: https://hdl.handle.net/1926/584 or http://www.insight-journal.org/browse/publication/176


    See:
     LabelToRGBImageFilter , LabelToRGBFunctor

     LabelMapOverlayImageFilter , LabelMapToBinaryImageFilter , LabelMapMaskImageFilter

     itk::simple::LabelMapToRGB for the procedural interface

     itk::LabelMapToRGBImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkLabelMapToRGBImageFilter.h

    
### sitk.LabelOverlapMeasuresImageFilter


    Computes overlap measures between the set same set of labels of pixels
    of two images. Background is assumed to be 0.


    This code was contributed in the Insight Journal paper: "Introducing
    Dice, Jaccard, and Other Label Overlap Measures To ITK" by Nicholas
    J. Tustison, James C. Gee https://hdl.handle.net/10380/3141 http://www.insight-journal.org/browse/publication/707


    Nicholas J. Tustison

    See:
     LabelOverlapMeasuresImageFilter

     itk::LabelOverlapMeasuresImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkLabelOverlapMeasuresImageFilter.h

    
### sitk.LabelOverlay
    LabelOverlay(Image image, Image labelImage, double opacity=0.5, double backgroundValue=0.0, VectorUInt8 colormap) -> Image



    Apply a colormap to a label image and put it on top of the input
    image.


    This function directly calls the execute method of LabelOverlayImageFilter in order to support a procedural API


    See:
     itk::simple::LabelOverlayImageFilter for the object oriented interface



    
### sitk.LabelOverlayImageFilter


    Apply a colormap to a label image and put it on top of the input
    image.


    Apply a colormap to a label image and put it on top of the input
    image. The set of colors is a good selection of distinct colors. The
    opacity of the label image can be defined by the user. The user can
    also choose if the want to use a background and which label value is
    the background. A background label produce a gray pixel with the same
    intensity than the input one.


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.
     This class was contributed to the Insight Journal https://hdl.handle.net/1926/172


    See:
     LabelToRGBImageFilter

     LabelMapOverlayImageFilter , LabelOverlayFunctor

     itk::simple::LabelOverlay for the procedural interface

     itk::LabelOverlayImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkLabelOverlayImageFilter.h

    
### sitk.LabelShapeStatisticsImageFilter


    Converts a label image to a label map and valuates the shape
    attributes.


    A convenient class that converts a label image to a label map and
    valuates the shape attribute at once.

    This implementation was taken from the Insight Journal paper:

    https://hdl.handle.net/1926/584 or http://www.insight-journal.org/browse/publication/176


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.

    See:
     ShapeLabelObject , LabelShapeOpeningImageFilter , LabelStatisticsOpeningImageFilter

     itk::LabelImageToShapeLabelMapFilter for the Doxygen on the original ITK class.


    C++ includes: sitkLabelShapeStatisticsImageFilter.h

    
### sitk.LabelStatisticsImageFilter


    Given an intensity image and a label map, compute min, max, variance
    and mean of the pixels associated with each label or segment.


    LabelStatisticsImageFilter computes the minimum, maximum, sum, mean, median, variance and sigma
    of regions of an intensity image, where the regions are defined via a
    label map (a second input). The label image should be integral type.
    The filter needs all of its input image. It behaves as a filter with
    an input and output. Thus it can be inserted in a pipline with other
    filters and the statistics will only be recomputed if a downstream
    filter changes.

    Optionally, the filter also computes intensity histograms on each
    object. If histograms are enabled, a median intensity value can also
    be computed, although its accuracy is limited to the bin width of the
    histogram. If histograms are not enabled, the median returns zero.

    The filter passes its intensity input through unmodified. The filter
    is threaded. It computes statistics in each thread then combines them
    in its AfterThreadedGenerate method.


    See:
     itk::LabelStatisticsImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkLabelStatisticsImageFilter.h

    
### sitk.LabelToRGB
    LabelToRGB(Image image1, double backgroundValue=0.0, VectorUInt8 colormap) -> Image



    Apply a colormap to a label image.


    This function directly calls the execute method of LabelToRGBImageFilter in order to support a procedural API


    See:
     itk::simple::LabelToRGBImageFilter for the object oriented interface



    
### sitk.LabelToRGBImageFilter


    Apply a colormap to a label image.


    Apply a colormap to a label image. The set of colors is a good
    selection of distinct colors. The user can choose to use a background
    value. In that case, a gray pixel with the same intensity than the
    background label is produced.

    This code was contributed in the Insight Journal paper: "The
    watershed transform in ITK - discussion and new developments" by
    Beare R., Lehmann G. https://hdl.handle.net/1926/202 http://www.insight-journal.org/browse/publication/92


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.

    Richard Beare. Department of Medicine, Monash University, Melbourne,
    Australia.

    See:
     LabelOverlayImageFilter

     LabelMapToRGBImageFilter , LabelToRGBFunctor, ScalarToRGBPixelFunctor

     itk::simple::LabelToRGB for the procedural interface

     itk::LabelToRGBImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkLabelToRGBImageFilter.h

    
### sitk.LabelUniqueLabelMap
    LabelUniqueLabelMap(Image image1, bool reverseOrdering=False) -> Image



    Make sure that the objects are not overlapping.


    This function directly calls the execute method of LabelUniqueLabelMapFilter in order to support a procedural API


    See:
     itk::simple::LabelUniqueLabelMapFilter for the object oriented interface



    
### sitk.LabelUniqueLabelMapFilter


    Make sure that the objects are not overlapping.


    AttributeUniqueLabelMapFilter search the overlapping zones in the overlapping objects and keeps
    only a single object on all the pixels of the image. The object to
    keep is selected according to their label.


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.
     This implementation was taken from the Insight Journal paper: https://hdl.handle.net/1926/584 or http://www.insight-journal.org/browse/publication/176


    See:
     AttributeLabelObject

     itk::simple::LabelUniqueLabelMapFilter for the procedural interface

     itk::LabelUniqueLabelMapFilter for the Doxygen on the original ITK class.


    C++ includes: sitkLabelUniqueLabelMapFilter.h

    
### sitk.LabelVoting
    LabelVoting(VectorOfImage images, uint64_t labelForUndecidedPixels) -> Image
    LabelVoting(Image image1, uint64_t labelForUndecidedPixels) -> Image
    LabelVoting(Image image1, Image image2, uint64_t labelForUndecidedPixels) -> Image
    LabelVoting(Image image1, Image image2, Image image3, uint64_t labelForUndecidedPixels) -> Image
    LabelVoting(Image image1, Image image2, Image image3, Image image4, uint64_t labelForUndecidedPixels) -> Image
    LabelVoting(Image image1, Image image2, Image image3, Image image4, Image image5, uint64_t labelForUndecidedPixels) -> Image
    
### sitk.LabelVotingImageFilter


    This filter performs pixelwise voting among an arbitrary number of
    input images, where each of them represents a segmentation of the same
    scene (i.e., image).


    Label voting is a simple method of classifier combination applied to
    image segmentation. Typically, the accuracy of the combined
    segmentation exceeds the accuracy of any of the input segmentations.
    Voting is therefore commonly used as a way of boosting segmentation
    performance.

    The use of label voting for combination of multiple segmentations is
    described in

    T. Rohlfing and C. R. Maurer, Jr., "Multi-classifier framework for
    atlas-based image segmentation," Pattern Recognition Letters, 2005.

    INPUTS
    All input volumes to this filter must be segmentations of an image,
    that is, they must have discrete pixel values where each value
    represents a different segmented object.
     Input volumes must all contain the same size RequestedRegions. Not all input images must contain all possible labels, but all label
    values must have the same meaning in all images.

    OUTPUTS
    The voting filter produces a single output volume. Each output pixel
    contains the label that occurred most often among the labels assigned
    to this pixel in all the input volumes, that is, the label that
    received the maximum number of "votes" from the input pixels.. If
    the maximum number of votes is not unique, i.e., if more than one
    label have a maximum number of votes, an "undecided" label is
    assigned to that output pixel.
     By default, the label used for undecided pixels is the maximum label
    value used in the input images plus one. Since it is possible for an
    image with 8 bit pixel values to use all 256 possible label values, it
    is permissible to combine 8 bit (i.e., byte) images into a 16 bit
    (i.e., short) output image.

    PARAMETERS
    The label used for "undecided" labels can be set using
    SetLabelForUndecidedPixels. This functionality can be unset by calling
    UnsetLabelForUndecidedPixels.

    Torsten Rohlfing, SRI International, Neuroscience Program

    See:
     itk::simple::LabelVoting for the procedural interface


    C++ includes: sitkLabelVotingImageFilter.h

    
### sitk.LandmarkBasedTransformInitializer
    LandmarkBasedTransformInitializer(Transform transform, VectorDouble fixedLandmarks, VectorDouble movingLandmarks, VectorDouble landmarkWeight, Image referenceImage, unsigned int numberOfControlPoints=4) -> Transform



    itk::simple::LandmarkBasedTransformInitializerFilter Procedural Interface


    This function directly calls the execute method of LandmarkBasedTransformInitializerFilter in order to support a procedural API


    See:
     itk::simple::LandmarkBasedTransformInitializerFilter for the object oriented interface



    
### sitk.LandmarkBasedTransformInitializerFilter


    This class computes the transform that aligns the fixed and moving
    images given a set of pair landmarks. The class is templated over the Transform type as well as fixed image and moving image types. The transform
    computed gives the best fit transform that maps the fixed and moving
    images in a least squares sense. The indices are taken to correspond,
    so point 1 in the first set will get mapped close to point 1 in the
    second set, etc.

    Currently, the following transforms are supported by the class: VersorRigid3DTransform Rigid2DTransform AffineTransform BSplineTransform

    An equal number of fixed and moving landmarks need to be specified
    using SetFixedLandmarks() and SetMovingLandmarks() . Any number of landmarks may be specified. In the case of using
    Affine or BSpline transforms, each landmark pair can contribute in the
    final transform based on its defined weight. Number of weights should
    be equal to the number of landmarks and can be specified using SetLandmarkWeight() . By defaults are weights are set to one. Call InitializeTransform()
    to initialize the transform.

    The class is based in part on Hybrid/vtkLandmarkTransform originally
    implemented in python by David G. Gobbi.

    The solution is based on Berthold K. P. Horn (1987), "Closed-form
    solution of absolute orientation using unit quaternions," http://people.csail.mit.edu/bkph/papers/Absolute_Orientation.pdf

    The Affine Transform initializer is based on an algorithm by H Spaeth, and is described in
    the Insight Journal Article "Affine Transformation for Landmark Based
    Registration Initializer in ITK" by Kim E.Y., Johnson H., Williams N.
    available at http://midasjournal.com/browse/publication/825

    Wiki Examples:

    All Examples

    Rigidly register one image to another using manually specified
    landmarks
    See:
     itk::simple::LandmarkBasedTransformInitializerFilter for the procedural interface

     itk::LandmarkBasedTransformInitializer for the Doxygen on the original ITK class.



    C++ includes: sitkLandmarkBasedTransformInitializerFilter.h

    
### sitk.LandweberDeconvolution
    LandweberDeconvolution(Image image1, Image image2, double alpha=0.1, int numberOfIterations=1, bool normalize=False, itk::simple::LandweberDeconvolutionImageFilter::BoundaryConditionType boundaryCondition, itk::simple::LandweberDeconvolutionImageFilter::OutputRegionModeType outputRegionMode) -> Image



    Deconvolve an image using the Landweber deconvolution algorithm.


    This function directly calls the execute method of LandweberDeconvolutionImageFilter in order to support a procedural API


    See:
     itk::simple::LandweberDeconvolutionImageFilter for the object oriented interface



    
### sitk.LandweberDeconvolutionImageFilter


    Deconvolve an image using the Landweber deconvolution algorithm.


    This filter implements the Landweber deconvolution algorthm as defined
    in Bertero M and Boccacci P, "Introduction to Inverse Problems in
    Imaging", 1998. The algorithm assumes that the input image has been
    formed by a linear shift-invariant system with a known kernel.

    The Landweber algorithm converges to a solution that minimizes the sum
    of squared errors $||f \otimes h - g||$ where $f$ is the estimate of the unblurred image, $\otimes$ is the convolution operator, $h$ is the blurring kernel, and $g$ is the blurred input image. As such, it is best suited for images
    that have zero-mean Gaussian white noise.

    This is the base implementation of the Landweber algorithm. It may
    produce results with negative values. For a version of this algorithm
    that enforces a positivity constraint on each intermediate solution,
    see ProjectedLandweberDeconvolutionImageFilter .

    This code was adapted from the Insight Journal contribution:

    "Deconvolution: infrastructure and reference algorithms" by Gaetan
    Lehmann https://hdl.handle.net/10380/3207


    Gaetan Lehmann, Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France

    Cory Quammen, The University of North Carolina at Chapel Hill

    See:
     IterativeDeconvolutionImageFilter

     RichardsonLucyDeconvolutionImageFilter

     ProjectedLandweberDeconvolutionImageFilter

     itk::simple::LandweberDeconvolution for the procedural interface

     itk::LandweberDeconvolutionImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkLandweberDeconvolutionImageFilter.h

    
### sitk.Laplacian
    Laplacian(Image image1, bool useImageSpacing=True) -> Image



    itk::simple::LaplacianImageFilter Procedural Interface


    This function directly calls the execute method of LaplacianImageFilter in order to support a procedural API


    See:
     itk::simple::LaplacianImageFilter for the object oriented interface



    
### sitk.LaplacianImageFilter


    This filter computes the Laplacian of a scalar-valued image. The
    Laplacian is an isotropic measure of the 2nd spatial derivative of an
    image. The Laplacian of an image highlights regions of rapid intensity
    change and is therefore often used for edge detection. Often, the
    Laplacian is applied to an image that has first been smoothed with a
    Gaussian filter in order to reduce its sensitivity to noise.


    The Laplacian at each pixel location is computed by convolution with
    the itk::LaplacianOperator .
    Inputs and Outputs
    The input to this filter is a scalar-valued itk::Image of arbitrary dimension. The output is a scalar-valued itk::Image .

    WARNING:
    The pixel type of the input and output images must be of real type
    (float or double). ConceptChecking is used here to enforce the input
    pixel type. You will get a compilation error if the pixel type of the
    input and output images is not float or double.

    See:
     Image

     Neighborhood

     NeighborhoodOperator

     NeighborhoodIterator

     LaplacianOperator

     itk::simple::Laplacian for the procedural interface

     itk::LaplacianImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkLaplacianImageFilter.h

    
### sitk.LaplacianRecursiveGaussian
    LaplacianRecursiveGaussian(Image image1, double sigma=1.0, bool normalizeAcrossScale=False) -> Image



    Computes the Laplacian of Gaussian (LoG) of an image.


    This function directly calls the execute method of LaplacianRecursiveGaussianImageFilter in order to support a procedural API


    See:
     itk::simple::LaplacianRecursiveGaussianImageFilter for the object oriented interface



    
### sitk.LaplacianRecursiveGaussianImageFilter


    Computes the Laplacian of Gaussian (LoG) of an image.


    Computes the Laplacian of Gaussian (LoG) of an image by convolution
    with the second derivative of a Gaussian. This filter is implemented
    using the recursive gaussian filters.
    See:
     itk::simple::LaplacianRecursiveGaussian for the procedural interface

     itk::LaplacianRecursiveGaussianImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkLaplacianRecursiveGaussianImageFilter.h

    
### sitk.LaplacianSegmentationLevelSet
    LaplacianSegmentationLevelSet(Image image1, Image image2, double maximumRMSError=0.02, double propagationScaling=1.0, double curvatureScaling=1.0, uint32_t numberOfIterations=1000, bool reverseExpansionDirection=False) -> Image



    Segments structures in images based on a second derivative image
    features.


    This function directly calls the execute method of LaplacianSegmentationLevelSetImageFilter in order to support a procedural API


    See:
     itk::simple::LaplacianSegmentationLevelSetImageFilter for the object oriented interface



    
### sitk.LaplacianSegmentationLevelSetImageFilter


    Segments structures in images based on a second derivative image
    features.


    IMPORTANT
    The SegmentationLevelSetImageFilter class and the LaplacianSegmentationLevelSetFunction class contain additional information necessary to the full
    understanding of how to use this filter.
    OVERVIEW
    This class is a level set method segmentation filter. It constructs a
    speed function which is zero at image edges as detected by a Laplacian
    filter. The evolving level set front will therefore tend to lock onto
    zero crossings in the image. The level set front moves fastest near
    edges.

    The Laplacian segmentation filter is intended primarily as a tool for
    refining existing segmentations. The initial isosurface (as given in
    the seed input image) should ideally be very close to the segmentation
    boundary of interest. The idea is that a rough segmentation can be
    refined by allowing the isosurface to deform slightly to achieve a
    better fit to the edge features of an image. One example of such an
    application is to refine the output of a hand segmented image.

    Because values in the Laplacian feature image will tend to be low
    except near edge features, this filter is not effective for segmenting
    large image regions from small seed surfaces.
    INPUTS
    This filter requires two inputs. The first input is a seed image. This
    seed image must contain an isosurface that you want to use as the seed
    for your segmentation. It can be a binary, graylevel, or floating
    point image. The only requirement is that it contain a closed
    isosurface that you will identify as the seed by setting the
    IsosurfaceValue parameter of the filter. For a binary image you will
    want to set your isosurface value halfway between your on and off
    values (i.e. for 0's and 1's, use an isosurface value of 0.5).

    The second input is the feature image. This is the image from which
    the speed function will be calculated. For most applications, this is
    the image that you want to segment. The desired isosurface in your
    seed image should lie within the region of your feature image that you
    are trying to segment.
     Note that this filter does no preprocessing of the feature image
    before thresholding. Because second derivative calculations are highly
    sensitive to noise, isotropic or anisotropic smoothing of the feature
    image can dramatically improve the results.


    See SegmentationLevelSetImageFilter for more information on Inputs.
    OUTPUTS
    The filter outputs a single, scalar, real-valued image. Positive
    *values in the output image are inside the segmentated region and
    negative *values in the image are outside of the inside region. The
    zero crossings of *the image correspond to the position of the level
    set front.

    See SparseFieldLevelSetImageFilter and SegmentationLevelSetImageFilter for more information.
    PARAMETERS
    This filter has no parameters other than those described in SegmentationLevelSetImageFilter .

    See:
     SegmentationLevelSetImageFilter

     LaplacianSegmentationLevelSetFunction ,

     SparseFieldLevelSetImageFilter

     itk::simple::LaplacianSegmentationLevelSet for the procedural interface

     itk::LaplacianSegmentationLevelSetImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkLaplacianSegmentationLevelSetImageFilter.h

    
### sitk.LaplacianSharpening
    LaplacianSharpening(Image image1, bool useImageSpacing=True) -> Image



    This filter sharpens an image using a Laplacian. LaplacianSharpening
    highlights regions of rapid intensity change and therefore highlights
    or enhances the edges. The result is an image that appears more in
    focus.


    This function directly calls the execute method of LaplacianSharpeningImageFilter in order to support a procedural API


    See:
     itk::simple::LaplacianSharpeningImageFilter for the object oriented interface



    
### sitk.LaplacianSharpeningImageFilter


    This filter sharpens an image using a Laplacian. LaplacianSharpening
    highlights regions of rapid intensity change and therefore highlights
    or enhances the edges. The result is an image that appears more in
    focus.


    The LaplacianSharpening at each pixel location is computed by
    convolution with the itk::LaplacianOperator .
    Inputs and Outputs
    The input to this filter is a scalar-valued itk::Image of arbitrary dimension. The output is a scalar-valued itk::Image .

    See:
     Image

     Neighborhood

     NeighborhoodOperator

     NeighborhoodIterator

     LaplacianOperator

     itk::simple::LaplacianSharpening for the procedural interface

     itk::LaplacianSharpeningImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkLaplacianSharpeningImageFilter.h

    
### sitk.Less
    Less(Image image1, Image image2, uint8_t backgroundValue=0, uint8_t foregroundValue=1) -> Image
    Less(Image image1, double constant, uint8_t backgroundValue=0, uint8_t foregroundValue=1) -> Image
    Less(double constant, Image image2, uint8_t backgroundValue=0, uint8_t foregroundValue=1) -> Image



    
### sitk.LessEqual
    LessEqual(Image image1, Image image2, uint8_t backgroundValue=0, uint8_t foregroundValue=1) -> Image
    LessEqual(Image image1, double constant, uint8_t backgroundValue=0, uint8_t foregroundValue=1) -> Image
    LessEqual(double constant, Image image2, uint8_t backgroundValue=0, uint8_t foregroundValue=1) -> Image



    
### sitk.LessEqualImageFilter


    Implements pixel-wise generic operation of two images, or of an image
    and a constant.


    This class is parameterized over the types of the two input images and
    the type of the output image. It is also parameterized by the
    operation to be applied. A Functor style is used.

    The constant must be of the same type than the pixel type of the
    corresponding image. It is wrapped in a SimpleDataObjectDecorator so it can be updated through the pipeline. The SetConstant() and
    GetConstant() methods are provided as shortcuts to set or get the
    constant value without manipulating the decorator.


    See:
     UnaryFunctorImageFilter TernaryFunctorImageFilter

     itk::simple::LessEqual for the procedural interface

     itk::BinaryFunctorImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkLessEqualImageFilter.h

    
### sitk.LessImageFilter


    Implements pixel-wise generic operation of two images, or of an image
    and a constant.


    This class is parameterized over the types of the two input images and
    the type of the output image. It is also parameterized by the
    operation to be applied. A Functor style is used.

    The constant must be of the same type than the pixel type of the
    corresponding image. It is wrapped in a SimpleDataObjectDecorator so it can be updated through the pipeline. The SetConstant() and
    GetConstant() methods are provided as shortcuts to set or get the
    constant value without manipulating the decorator.


    See:
     UnaryFunctorImageFilter TernaryFunctorImageFilter

     itk::simple::Less for the procedural interface

     itk::BinaryFunctorImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkLessImageFilter.h

    
### sitk.LevelSetMotionRegistrationFilter


    Deformably register two images using level set motion.


    LevelSetMotionFilter implements a deformable registration algorithm
    that aligns a fixed and a moving image under level set motion. The
    equations of motion are similar to those of the DemonsRegistrationFilter . The main differences are: (1) Gradients of the moving image are
    calculated on a smoothed image while intensity difference are measured
    on the original images (2) Magnitude of the motion vector is a
    function of the differences in intensity between the fixed and moving
    pixel. An adaptive timestep is calculated based on the maximum motion
    vector over the entire field to ensure stability. The timestep also
    implictly converts the motion vector measured in units of intensity to
    a vector measured in physical units. Demons, on the other hand,
    defines its motion vectors as function of both the intensity
    differences and gradient magnitude at each respective pixel. Consider
    two separate pixels with the same intensity differences between the
    corresponding fixed and moving pixel pairs. In demons, the motion
    vector of the pixel over a low gradient region will be larger than the
    motion vector of the pixel over a large gradient region. This leads to
    an unstable vector field. In the levelset approach, the motion vectors
    will be proportional to the gradients, scaled by the maximum gradient
    over the entire field. The pixel with at the lower gradient position
    will more less than the pixel at the higher gradient position. (3)
    Gradients are calculated using minmod finite difference instead of
    using central differences.

    A deformation field is represented as a image whose pixel type is some
    vector type with at least N elements, where N is the dimension of the
    fixed image. The vector type must support element access via operator
    []. It is assumed that the vector elements behave like floating point
    scalars.

    This class is templated over the fixed image type, moving image type
    and the deformation field type.

    The input fixed and moving images are set via methods SetFixedImage
    and SetMovingImage respectively. An initial deformation field maybe
    set via SetInitialDisplacementField or SetInput. If no initial field
    is set, a zero field is used as the initial condition.

    The algorithm has one parameters: the number of iteration to be
    performed.

    The output deformation field can be obtained via methods GetOutput or
    GetDisplacementField.

    This class make use of the finite difference solver hierarchy. Update
    for each iteration is computed in LevelSetMotionFunction.


    WARNING:
    This filter assumes that the fixed image type, moving image type and
    deformation field type all have the same number of dimensions.
     Ref: B.C. Vemuri, J. Ye, Y. Chen, C.M. Leonard. "Image registration
    via level-set motion: applications to atlas-based segmentation".
    Medical Image Analysis. Vol. 7. pp. 1-20. 2003.


    See:
     LevelSetMotionRegistrationFunction

     DemonsRegistrationFilter

     itk::LevelSetMotionRegistrationFilter for the Doxygen on the original ITK class.


    C++ includes: sitkLevelSetMotionRegistrationFilter.h

    
### sitk.LiThreshold
    LiThreshold(Image image, Image maskImage, uint8_t insideValue=1, uint8_t outsideValue=0, uint32_t numberOfHistogramBins=256, bool maskOutput=True, uint8_t maskValue=255) -> Image
    LiThreshold(Image image, uint8_t insideValue=1, uint8_t outsideValue=0, uint32_t numberOfHistogramBins=256, bool maskOutput=True, uint8_t maskValue=255) -> Image



    
### sitk.LiThresholdImageFilter


    Threshold an image using the Li Threshold.


    This filter creates a binary thresholded image that separates an image
    into foreground and background components. The filter computes the
    threshold using the LiThresholdCalculator and applies that threshold to the input image using the BinaryThresholdImageFilter .


    Richard Beare. Department of Medicine, Monash University, Melbourne,
    Australia.

    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.
     This implementation was taken from the Insight Journal paper: https://hdl.handle.net/10380/3279 or http://www.insight-journal.org/browse/publication/811


    See:
     HistogramThresholdImageFilter

     itk::simple::LiThreshold for the procedural interface

     itk::LiThresholdImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkLiThresholdImageFilter.h

    
### sitk.Log
    Log(Image image1) -> Image



    Computes the log() of each pixel.


    This function directly calls the execute method of LogImageFilter in order to support a procedural API


    See:
     itk::simple::LogImageFilter for the object oriented interface



    
### sitk.Log10
    Log10(Image image1) -> Image



    Computes the log10 of each pixel.


    This function directly calls the execute method of Log10ImageFilter in order to support a procedural API


    See:
     itk::simple::Log10ImageFilter for the object oriented interface



    
### sitk.Log10ImageFilter


    Computes the log10 of each pixel.


    The computation is performed using std::log10(x).
    See:
     itk::simple::Log10 for the procedural interface

     itk::Log10ImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkLog10ImageFilter.h

    
### sitk.LogImageFilter


    Computes the log() of each pixel.



    See:
     itk::simple::Log for the procedural interface

     itk::LogImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkLogImageFilter.h

    
### sitk.MagnitudeAndPhaseToComplex
    MagnitudeAndPhaseToComplex(Image image1, Image image2) -> Image
    MagnitudeAndPhaseToComplex(Image image1, double constant) -> Image
    MagnitudeAndPhaseToComplex(double constant, Image image2) -> Image



    
### sitk.MagnitudeAndPhaseToComplexImageFilter


    Implements pixel-wise conversion of magnitude and phase data into
    complex voxels.


    This filter is parametrized over the types of the two input images and
    the type of the output image.

    The filter expect all images to have the same dimension (e.g. all 2D,
    or all 3D, or all ND)
    See:
     itk::simple::MagnitudeAndPhaseToComplex for the procedural interface

     itk::MagnitudeAndPhaseToComplexImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkMagnitudeAndPhaseToComplexImageFilter.h

    
### sitk.Mask
    Mask(Image image, Image maskImage, double outsideValue=0, double maskingValue=0) -> Image



    Mask an image with a mask.


    This function directly calls the execute method of MaskImageFilter in order to support a procedural API


    See:
     itk::simple::MaskImageFilter for the object oriented interface



    
### sitk.MaskImageFilter


    Mask an image with a mask.


    This class is templated over the types of the input image type, the
    mask image type and the type of the output image. Numeric conversions
    (castings) are done by the C++ defaults.

    The pixel type of the input 2 image must have a valid definition of
    the operator != with zero. This condition is required because
    internally this filter will perform the operation


    The pixel from the input 1 is cast to the pixel type of the output
    image.

    Note that the input and the mask images must be of the same size.


    WARNING:
    Any pixel value other than masking value (0 by default) will not be
    masked out.

    See:
     MaskNegatedImageFilter

     itk::simple::Mask for the procedural interface

     itk::MaskImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkMaskImageFilter.h

    
### sitk.MaskNegated
    MaskNegated(Image image, Image maskImage, double outsideValue=0, double maskingValue=0) -> Image



    Mask an image with the negation (or logical compliment) of a mask.


    This function directly calls the execute method of MaskNegatedImageFilter in order to support a procedural API


    See:
     itk::simple::MaskNegatedImageFilter for the object oriented interface



    
### sitk.MaskNegatedImageFilter


    Mask an image with the negation (or logical compliment) of a mask.


    This class is templated over the types of the input image type, the
    mask image type and the type of the output image. Numeric conversions
    (castings) are done by the C++ defaults.

    The pixel type of the input 2 image must have a valid definition of
    the operator!=. This condition is required because internally this
    filter will perform the operation


    The pixel from the input 1 is cast to the pixel type of the output
    image.

    Note that the input and the mask images must be of the same size.


    WARNING:
    Only pixel value with mask_value ( defaults to 0 ) will be preserved.

    See:
     MaskImageFilter

     itk::simple::MaskNegated for the procedural interface

     itk::MaskNegatedImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkMaskNegatedImageFilter.h

    
### sitk.MaskedFFTNormalizedCorrelation
    MaskedFFTNormalizedCorrelation(Image fixedImage, Image movingImage, Image fixedImageMask, Image movingImageMask, uint64_t requiredNumberOfOverlappingPixels=0, float requiredFractionOfOverlappingPixels=0.0) -> Image



    Calculate masked normalized cross correlation using FFTs.


    This function directly calls the execute method of MaskedFFTNormalizedCorrelationImageFilter in order to support a procedural API


    See:
     itk::simple::MaskedFFTNormalizedCorrelationImageFilter for the object oriented interface



    
### sitk.MaskedFFTNormalizedCorrelationImageFilter


    Calculate masked normalized cross correlation using FFTs.


    This filter calculates the masked normalized cross correlation (NCC)
    of two images under masks using FFTs instead of spatial correlation.
    It is much faster than spatial correlation for reasonably large
    structuring elements. This filter is not equivalent to simply masking
    the images first and then correlating them; the latter approach yields
    incorrect results because the zeros in the images still affect the
    metric in the correlation process. This filter implements the masked
    NCC correctly so that the masked-out regions are completely ignored.
    The fundamental difference is described in detail in the references
    below. If the masks are set to images of all ones, the result of this
    filter is the same as standard NCC.

    Inputs: Two images are required as inputs, fixedImage and movingImage,
    and two are optional, fixedMask and movingMask. In the context of
    correlation, inputs are often defined as: "image" and "template".
    In this filter, the fixedImage plays the role of the image, and the
    movingImage plays the role of the template. However, this filter is
    capable of correlating any two images and is not restricted to small
    movingImages (templates). In the fixedMask and movingMask, non-zero
    positive values indicate locations of useful information in the
    corresponding image, whereas zero and negative values indicate
    locations that should be masked out (ignored). Internally, the masks
    are converted to have values of only 0 and 1. For each optional mask
    that is not set, the filter internally creates an image of ones, which
    is equivalent to not masking the image. Thus, if both masks are not
    set, the result will be equivalent to unmasked NCC. For example, if
    only a mask for the fixed image is needed, the movingMask can either
    not be set or can be set to an image of ones.

    Optional parameters: The RequiredNumberOfOverlappingPixels enables the
    user to specify the minimum number of voxels of the two masks that
    must overlap; any location in the correlation map that results from
    fewer than this number of voxels will be set to zero. Larger values
    zero-out pixels on a larger border around the correlation image. Thus,
    larger values remove less stable computations but also limit the
    capture range. If RequiredNumberOfOverlappingPixels is set to 0, the
    default, no zeroing will take place.

    The RequiredFractionOfOverlappingPixels enables the user to specify a
    fraction of the maximum number of overlapping pixels that need to
    overlap; any location in the correlation map that results from fewer
    than the product of this fraction and the internally computed maximum
    number of overlapping pixels will be set to zero. The value ranges
    between 0.0 and 1.0. This is very useful when the user does does not
    know beforehand the maximum number of pixels of the masks that will
    overlap. For example, when the masks have strange shapes, it is
    difficult to predict how the correlation of the masks will interact
    and what the maximum overlap will be. It is also useful when the mask
    shapes or sizes change because it is relative to the internally
    computed maximum of the overlap. Larger values zero-out pixels on a
    larger border around the correlation image. Thus, larger values remove
    less stable computations but also limit the capture range. Experiments
    have shown that a value between 0.1 and 0.6 works well for images with
    significant overlap and between 0.05 and 0.1 for images with little
    overlap (such as in stitching applications). If
    RequiredFractionOfOverlappingPixels is set to 0, the default, no
    zeroing will take place.

    The user can either specify RequiredNumberOfOverlappingPixels or
    RequiredFractionOfOverlappingPixels (or both or none). Internally, the
    number of required pixels resulting from both of these methods is
    calculated and the one that gives the largest number of pixels is
    chosen. Since these both default to 0, if a user only sets one, the
    other is ignored.

    Image size: fixedImage and movingImage need not be the same size, but
    fixedMask must be the same size as fixedImage, and movingMask must be
    the same size as movingImage. Furthermore, whereas some algorithms
    require that the "template" be smaller than the "image" because of
    errors in the regions where the two are not fully overlapping, this
    filter has no such restriction.

    Image spacing: Since the computations are done in the pixel domain, all
    input images must have the same spacing.

    Outputs; The output is an image of RealPixelType that is the masked
    NCC of the two images and its values range from -1.0 to 1.0. The size
    of this NCC image is, by definition, size(fixedImage) +
    size(movingImage) - 1.

    Example filter usage:


    WARNING:
    The pixel type of the output image must be of real type (float or
    double). ConceptChecking is used to enforce the output pixel type. You
    will get a compilation error if the pixel type of the output image is
    not float or double.
     References: 1) D. Padfield. "Masked object registration in the
    Fourier domain." Transactions on Image Processing. 2) D. Padfield. "Masked FFT registration". In Proc.
    Computer Vision and Pattern Recognition, 2010.


    : Dirk Padfield, GE Global Research, padfield@research.ge.com

    See:
     itk::simple::MaskedFFTNormalizedCorrelation for the procedural interface

     itk::MaskedFFTNormalizedCorrelationImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkMaskedFFTNormalizedCorrelationImageFilter.h

    
### sitk.Maximum
    Maximum(Image image1, Image image2) -> Image
    Maximum(Image image1, double constant) -> Image
    Maximum(double constant, Image image2) -> Image



    
### sitk.MaximumEntropyThreshold
    MaximumEntropyThreshold(Image image, Image maskImage, uint8_t insideValue=1, uint8_t outsideValue=0, uint32_t numberOfHistogramBins=256, bool maskOutput=True, uint8_t maskValue=255) -> Image
    MaximumEntropyThreshold(Image image, uint8_t insideValue=1, uint8_t outsideValue=0, uint32_t numberOfHistogramBins=256, bool maskOutput=True, uint8_t maskValue=255) -> Image



    
### sitk.MaximumEntropyThresholdImageFilter


    Threshold an image using the MaximumEntropy Threshold.


    This filter creates a binary thresholded image that separates an image
    into foreground and background components. The filter computes the
    threshold using the MaximumEntropyThresholdCalculator and applies that threshold to the input image using the BinaryThresholdImageFilter .


    Richard Beare. Department of Medicine, Monash University, Melbourne,
    Australia.

    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.
     This implementation was taken from the Insight Journal paper: https://hdl.handle.net/10380/3279 or http://www.insight-journal.org/browse/publication/811


    See:
     HistogramThresholdImageFilter

     itk::simple::MaximumEntropyThreshold for the procedural interface

     itk::MaximumEntropyThresholdImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkMaximumEntropyThresholdImageFilter.h

    
### sitk.MaximumImageFilter


    Implements a pixel-wise operator Max(a,b) between two images.


    The pixel values of the output image are the maximum between the
    corresponding pixels of the two input images.

    This class is templated over the types of the two input images and the
    type of the output image. Numeric conversions (castings) are done by
    the C++ defaults.
    See:
     itk::simple::Maximum for the procedural interface

     itk::MaximumImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkMaximumImageFilter.h

    
### sitk.MaximumProjection
    MaximumProjection(Image image1, unsigned int projectionDimension=0) -> Image



    Maximum projection.


    This function directly calls the execute method of MaximumProjectionImageFilter in order to support a procedural API


    See:
     itk::simple::MaximumProjectionImageFilter for the object oriented interface



    
### sitk.MaximumProjectionImageFilter


    Maximum projection.


    This class was contributed to the insight journal by Gaetan Lehmann.
    The original paper can be found at https://hdl.handle.net/1926/164


    Gaetan Lehmann. Biologie du Developpement et de la reproduction, inra
    de jouy-en-josas, France.

    See:
     ProjectionImageFilter

     MedianProjectionImageFilter

     MeanProjectionImageFilter

     MinimumProjectionImageFilter

     StandardDeviationProjectionImageFilter

     SumProjectionImageFilter

     BinaryProjectionImageFilter

     itk::simple::MaximumProjection for the procedural interface

     itk::MaximumProjectionImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkMaximumProjectionImageFilter.h

    
### sitk.Mean
    Mean(Image image1, VectorUInt32 radius) -> Image



    Applies an averaging filter to an image.


    This function directly calls the execute method of MeanImageFilter in order to support a procedural API


    See:
     itk::simple::MeanImageFilter for the object oriented interface



    
### sitk.MeanImageFilter


    Applies an averaging filter to an image.


    Computes an image where a given pixel is the mean value of the the
    pixels in a neighborhood about the corresponding input pixel.

    A mean filter is one of the family of linear filters.


    See:
     Image

     Neighborhood

     NeighborhoodOperator

     NeighborhoodIterator

     itk::simple::Mean for the procedural interface

     itk::MeanImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkMeanImageFilter.h

    
### sitk.MeanProjection
    MeanProjection(Image image1, unsigned int projectionDimension=0) -> Image



    Mean projection.


    This function directly calls the execute method of MeanProjectionImageFilter in order to support a procedural API


    See:
     itk::simple::MeanProjectionImageFilter for the object oriented interface



    
### sitk.MeanProjectionImageFilter


    Mean projection.


    This class was contributed to the Insight Journal by Gaetan Lehmann.
    The original paper can be found at https://hdl.handle.net/1926/164


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.

    See:
     ProjectionImageFilter

     MedianProjectionImageFilter

     MinimumProjectionImageFilter

     StandardDeviationProjectionImageFilter

     SumProjectionImageFilter

     BinaryProjectionImageFilter

     MaximumProjectionImageFilter

     itk::simple::MeanProjection for the procedural interface

     itk::MeanProjectionImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkMeanProjectionImageFilter.h

    
### sitk.Median
    Median(Image image1, VectorUInt32 radius) -> Image



    Applies a median filter to an image.


    This function directly calls the execute method of MedianImageFilter in order to support a procedural API


    See:
     itk::simple::MedianImageFilter for the object oriented interface



    
### sitk.MedianImageFilter


    Applies a median filter to an image.


    Computes an image where a given pixel is the median value of the the
    pixels in a neighborhood about the corresponding input pixel.

    A median filter is one of the family of nonlinear filters. It is used
    to smooth an image without being biased by outliers or shot noise.

    This filter requires that the input pixel type provides an operator<() (LessThan Comparable).


    See:
     Image

     Neighborhood

     NeighborhoodOperator

     NeighborhoodIterator

     itk::simple::Median for the procedural interface

     itk::MedianImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkMedianImageFilter.h

    
### sitk.MedianProjection
    MedianProjection(Image image1, unsigned int projectionDimension=0) -> Image



    Median projection.


    This function directly calls the execute method of MedianProjectionImageFilter in order to support a procedural API


    See:
     itk::simple::MedianProjectionImageFilter for the object oriented interface



    
### sitk.MedianProjectionImageFilter


    Median projection.


    This class was contributed to the Insight Journal by Gaetan Lehmann.
    The original paper can be found at https://hdl.handle.net/1926/164


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.

    See:
     ProjectionImageFilter

     StandardDeviationProjectionImageFilter

     SumProjectionImageFilter

     BinaryProjectionImageFilter

     MaximumProjectionImageFilter

     MinimumProjectionImageFilter

     MeanProjectionImageFilter

     itk::simple::MedianProjection for the procedural interface

     itk::MedianProjectionImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkMedianProjectionImageFilter.h

    
### sitk.MergeLabelMap
    MergeLabelMap(VectorOfImage images, itk::simple::MergeLabelMapFilter::MethodType method) -> Image
    MergeLabelMap(Image image1, itk::simple::MergeLabelMapFilter::MethodType method) -> Image
    MergeLabelMap(Image image1, Image image2, itk::simple::MergeLabelMapFilter::MethodType method) -> Image
    MergeLabelMap(Image image1, Image image2, Image image3, itk::simple::MergeLabelMapFilter::MethodType method) -> Image
    MergeLabelMap(Image image1, Image image2, Image image3, Image image4, itk::simple::MergeLabelMapFilter::MethodType method) -> Image
    MergeLabelMap(Image image1, Image image2, Image image3, Image image4, Image image5, itk::simple::MergeLabelMapFilter::MethodType method) -> Image
    
### sitk.MergeLabelMapFilter


    Merges several Label Maps.


    This filter takes one or more input Label Map and merges them.

    SetMethod() can be used to change how the filter manage the labels from the
    different label maps. KEEP (0): MergeLabelMapFilter do its best to keep the label unchanged, but if a label is already
    used in a previous label map, a new label is assigned. AGGREGATE (1):
    If the same label is found several times in the label maps, the label
    objects with the same label are merged. PACK (2): MergeLabelMapFilter relabel all the label objects by order of processing. No conflict can
    occur. STRICT (3): MergeLabelMapFilter keeps the labels unchanged and raises an exception if the same label
    is found in several images.

    This implementation was taken from the Insight Journal paper: https://hdl.handle.net/1926/584 or http://www.insight-journal.org/browse/publication/176


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.

    See:
     ShapeLabelObject , RelabelComponentImageFilter

     itk::simple::MergeLabelMapFilter for the procedural interface


    C++ includes: sitkMergeLabelMapFilter.h

    
### sitk.MinMaxCurvatureFlow
    MinMaxCurvatureFlow(Image image1, double timeStep=0.05, uint32_t numberOfIterations=5, int stencilRadius=2) -> Image



    Denoise an image using min/max curvature flow.


    This function directly calls the execute method of MinMaxCurvatureFlowImageFilter in order to support a procedural API


    See:
     itk::simple::MinMaxCurvatureFlowImageFilter for the object oriented interface



    
### sitk.MinMaxCurvatureFlowImageFilter


    Denoise an image using min/max curvature flow.


    MinMaxCurvatureFlowImageFilter implements a curvature driven image denoising algorithm. Iso-
    brightness contours in the grayscale input image are viewed as a level
    set. The level set is then evolved using a curvature-based speed
    function:

    \[ I_t = F_{\mbox{minmax}} |\nabla I| \]

    where $ F_{\mbox{minmax}} = \max(\kappa,0) $ if $ \mbox{Avg}_{\mbox{stencil}}(x) $ is less than or equal to $ T_{thresold} $ and $ \min(\kappa,0) $ , otherwise. $ \kappa $ is the mean curvature of the iso-brightness contour at point $ x $ .

    In min/max curvature flow, movement is turned on or off depending on
    the scale of the noise one wants to remove. Switching depends on the
    average image value of a region of radius $ R $ around each point. The choice of $ R $ , the stencil radius, governs the scale of the noise to be removed.

    The threshold value $ T_{threshold} $ is the average intensity obtained in the direction perpendicular to
    the gradient at point $ x $ at the extrema of the local neighborhood.

    This filter make use of the multi-threaded finite difference solver
    hierarchy. Updates are computed using a MinMaxCurvatureFlowFunction object. A zero flux Neumann boundary condition is used when computing
    derivatives near the data boundary.


    WARNING:
    This filter assumes that the input and output types have the same
    dimensions. This filter also requires that the output image pixels are
    of a real type. This filter works for any dimensional images, however
    for dimensions greater than 3D, an expensive brute-force search is
    used to compute the local threshold.
     Reference: "Level Set Methods and Fast Marching Methods", J.A.
    Sethian, Cambridge Press, Chapter 16, Second edition, 1999.


    See:
     MinMaxCurvatureFlowFunction

     CurvatureFlowImageFilter

     BinaryMinMaxCurvatureFlowImageFilter

     itk::simple::MinMaxCurvatureFlow for the procedural interface

     itk::MinMaxCurvatureFlowImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkMinMaxCurvatureFlowImageFilter.h

    
### sitk.Minimum
    Minimum(Image image1, Image image2) -> Image
    Minimum(Image image1, double constant) -> Image
    Minimum(double constant, Image image2) -> Image



    
### sitk.MinimumImageFilter


    Implements a pixel-wise operator Min(a,b) between two images.


    The pixel values of the output image are the minimum between the
    corresponding pixels of the two input images.

    This class is templated over the types of the two input images and the
    type of the output image. Numeric conversions (castings) are done by
    the C++ defaults.
    See:
     itk::simple::Minimum for the procedural interface

     itk::MinimumImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkMinimumImageFilter.h

    
### sitk.MinimumMaximumImageFilter


    Computes the minimum and the maximum intensity values of an image.


    It is templated over input image type only. This filter just copies
    the input image through this output to be included within the
    pipeline. The implementation uses the StatisticsImageFilter .


    See:
     StatisticsImageFilter

     itk::MinimumMaximumImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkMinimumMaximumImageFilter.h

    
### sitk.MinimumProjection
    MinimumProjection(Image image1, unsigned int projectionDimension=0) -> Image



    Minimum projection.


    This function directly calls the execute method of MinimumProjectionImageFilter in order to support a procedural API


    See:
     itk::simple::MinimumProjectionImageFilter for the object oriented interface



    
### sitk.MinimumProjectionImageFilter


    Minimum projection.


    This class was contributed to the Insight Journal by Gaetan Lehmann.
    The original paper can be found at https://hdl.handle.net/1926/164


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.

    See:
     ProjectionImageFilter

     StandardDeviationProjectionImageFilter

     SumProjectionImageFilter

     BinaryProjectionImageFilter

     MaximumProjectionImageFilter

     MeanProjectionImageFilter

     itk::simple::MinimumProjection for the procedural interface

     itk::MinimumProjectionImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkMinimumProjectionImageFilter.h

    
### sitk.MirrorPad
    MirrorPad(Image image1, VectorUInt32 padLowerBound, VectorUInt32 padUpperBound) -> Image



    Increase the image size by padding with replicants of the input image
    value.


    This function directly calls the execute method of MirrorPadImageFilter in order to support a procedural API


    See:
     itk::simple::MirrorPadImageFilter for the object oriented interface



    
### sitk.MirrorPadImageFilter


    Increase the image size by padding with replicants of the input image
    value.


    MirrorPadImageFilter changes the image bounds of an image. Any added pixels are filled in
    with a mirrored replica of the input image. For instance, if the
    output image needs a pixel that is two pixels to the left of the
    LargestPossibleRegion of the input image, the value assigned will be
    from the pixel two pixels inside the left boundary of the
    LargestPossibleRegion. The image bounds of the output must be
    specified.

    Visual explanation of padding regions. This filter is implemented as a
    multithreaded filter. It provides a ThreadedGenerateData() method for
    its implementation.


    See:
     WrapPadImageFilter , ConstantPadImageFilter

     itk::simple::MirrorPad for the procedural interface

     itk::MirrorPadImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkMirrorPadImageFilter.h

    
### sitk.Modulus
    Modulus(Image image1, Image image2) -> Image
    Modulus(Image image1, uint32_t constant) -> Image
    Modulus(uint32_t constant, Image image2) -> Image



    
### sitk.ModulusImageFilter


    Computes the modulus (x % dividend) pixel-wise.


    The input pixel type must support the c++ modulus operator (%).

    If the dividend is zero, the maximum value will be returned.


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.

    See:
     itk::simple::Modulus for the procedural interface

     itk::ModulusImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkModulusImageFilter.h

    
### sitk.MomentsThreshold
    MomentsThreshold(Image image, Image maskImage, uint8_t insideValue=1, uint8_t outsideValue=0, uint32_t numberOfHistogramBins=256, bool maskOutput=True, uint8_t maskValue=255) -> Image
    MomentsThreshold(Image image, uint8_t insideValue=1, uint8_t outsideValue=0, uint32_t numberOfHistogramBins=256, bool maskOutput=True, uint8_t maskValue=255) -> Image



    
### sitk.MomentsThresholdImageFilter


    Threshold an image using the Moments Threshold.


    This filter creates a binary thresholded image that separates an image
    into foreground and background components. The filter computes the
    threshold using the MomentsThresholdCalculator and applies that threshold to the input image using the BinaryThresholdImageFilter .


    Richard Beare. Department of Medicine, Monash University, Melbourne,
    Australia.

    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.
     This implementation was taken from the Insight Journal paper: https://hdl.handle.net/10380/3279 or http://www.insight-journal.org/browse/publication/811


    See:
     HistogramThresholdImageFilter

     itk::simple::MomentsThreshold for the procedural interface

     itk::MomentsThresholdImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkMomentsThresholdImageFilter.h

    
### sitk.MorphologicalGradient
    MorphologicalGradient(Image arg1, uint32_t radius=1, itk::simple::KernelEnum kernel) -> Image
    MorphologicalGradient(Image arg1, VectorUInt32 vectorRadius, itk::simple::KernelEnum kernel) -> Image



    itk::simple::MorphologicalGradientImageFilter Functional Interface

    This function directly calls the execute method of MorphologicalGradientImageFilter in order to support a fully functional API


    
### sitk.MorphologicalGradientImageFilter


    gray scale dilation of an image


    Dilate an image using grayscale morphology. Dilation takes the maximum
    of all the pixels identified by the structuring element.

    The structuring element is assumed to be composed of binary values
    (zero or one). Only elements of the structuring element having values
    > 0 are candidates for affecting the center pixel.


    See:
     MorphologyImageFilter , GrayscaleFunctionDilateImageFilter , BinaryDilateImageFilter

     itk::simple::MorphologicalGradient for the procedural interface

     itk::MorphologicalGradientImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkMorphologicalGradientImageFilter.h

    
### sitk.MorphologicalWatershed
    MorphologicalWatershed(Image image1, double level=0.0, bool markWatershedLine=True, bool fullyConnected=False) -> Image



    Watershed segmentation implementation with morphogical operators.


    This function directly calls the execute method of MorphologicalWatershedImageFilter in order to support a procedural API


    See:
     itk::simple::MorphologicalWatershedImageFilter for the object oriented interface



    
### sitk.MorphologicalWatershedFromMarkers
    MorphologicalWatershedFromMarkers(Image image, Image markerImage, bool markWatershedLine=True, bool fullyConnected=False) -> Image



    Morphological watershed transform from markers.


    This function directly calls the execute method of MorphologicalWatershedFromMarkersImageFilter in order to support a procedural API


    See:
     itk::simple::MorphologicalWatershedFromMarkersImageFilter for the object oriented interface



    
### sitk.MorphologicalWatershedFromMarkersImageFilter


    Morphological watershed transform from markers.


    The watershed transform is a tool for image segmentation that is fast
    and flexible and potentially fairly parameter free. It was originally
    derived from a geophysical model of rain falling on a terrain and a
    variety of more formal definitions have been devised to allow
    development of practical algorithms. If an image is considered as a
    terrain and divided into catchment basins then the hope is that each
    catchment basin would contain an object of interest.

    The output is a label image. A label image, sometimes referred to as a
    categorical image, has unique values for each region. For example, if
    a watershed produces 2 regions, all pixels belonging to one region
    would have value A, and all belonging to the other might have value B.
    Unassigned pixels, such as watershed lines, might have the background
    value (0 by convention).

    The simplest way of using the watershed is to preprocess the image we
    want to segment so that the boundaries of our objects are bright (e.g
    apply an edge detector) and compute the watershed transform of the
    edge image. Watershed lines will correspond to the boundaries and our
    problem will be solved. This is rarely useful in practice because
    there are always more regional minima than there are objects, either
    due to noise or natural variations in the object surfaces. Therefore,
    while many watershed lines do lie on significant boundaries, there are
    many that don't. Various methods can be used to reduce the number of
    minima in the image, like thresholding the smallest values, filtering
    the minima and/or smoothing the image.

    This filter use another approach to avoid the problem of over
    segmentation: it let the user provide a marker image which mark the
    minima in the input image and give them a label. The minima are
    imposed in the input image by the markers. The labels of the output
    image are the label of the marker image.

    The morphological watershed transform algorithm is described in
    Chapter 9.2 of Pierre Soille's book "Morphological Image Analysis:
    Principles and Applications", Second Edition, Springer, 2003.

    This code was contributed in the Insight Journal paper: "The
    watershed transform in ITK - discussion and new developments" by
    Beare R., Lehmann G. https://hdl.handle.net/1926/202 http://www.insight-journal.org/browse/publication/92


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.

    Richard Beare. Department of Medicine, Monash University, Melbourne,
    Australia.

    See:
     WatershedImageFilter , MorphologicalWatershedImageFilter

     itk::simple::MorphologicalWatershedFromMarkers for the procedural interface

     itk::MorphologicalWatershedFromMarkersImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkMorphologicalWatershedFromMarkersImageFilter.h

    
### sitk.MorphologicalWatershedImageFilter


    Watershed segmentation implementation with morphogical operators.


    Watershed pixel are labeled 0. TOutputImage should be an integer type.
    Labels of output image are in no particular order. You can reorder the
    labels such that object labels are consecutive and sorted based on
    object size by passing the output of this filter to a RelabelComponentImageFilter .

    The morphological watershed transform algorithm is described in
    Chapter 9.2 of Pierre Soille's book "Morphological Image Analysis:
    Principles and Applications", Second Edition, Springer, 2003.

    This code was contributed in the Insight Journal paper: "The
    watershed transform in ITK - discussion and new developments" by
    Beare R., Lehmann G. https://hdl.handle.net/1926/202 http://www.insight-journal.org/browse/publication/92


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.

    See:
     WatershedImageFilter , MorphologicalWatershedFromMarkersImageFilter

     itk::simple::MorphologicalWatershed for the procedural interface

     itk::MorphologicalWatershedImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkMorphologicalWatershedImageFilter.h

    
### sitk.MultiLabelSTAPLE
    MultiLabelSTAPLE(VectorOfImage images, uint64_t labelForUndecidedPixels, float terminationUpdateThreshold=1e-5, unsigned int maximumNumberOfIterations, VectorFloat priorProbabilities) -> Image
    MultiLabelSTAPLE(Image image1, uint64_t labelForUndecidedPixels, float terminationUpdateThreshold=1e-5, unsigned int maximumNumberOfIterations, VectorFloat priorProbabilities) -> Image
    MultiLabelSTAPLE(Image image1, Image image2, uint64_t labelForUndecidedPixels, float terminationUpdateThreshold=1e-5, unsigned int maximumNumberOfIterations, VectorFloat priorProbabilities) -> Image
    MultiLabelSTAPLE(Image image1, Image image2, Image image3, uint64_t labelForUndecidedPixels, float terminationUpdateThreshold=1e-5, unsigned int maximumNumberOfIterations, VectorFloat priorProbabilities) -> Image
    MultiLabelSTAPLE(Image image1, Image image2, Image image3, Image image4, uint64_t labelForUndecidedPixels, float terminationUpdateThreshold=1e-5, unsigned int maximumNumberOfIterations, VectorFloat priorProbabilities) -> Image
    MultiLabelSTAPLE(Image image1, Image image2, Image image3, Image image4, Image image5, uint64_t labelForUndecidedPixels, float terminationUpdateThreshold=1e-5, unsigned int maximumNumberOfIterations, VectorFloat priorProbabilities) -> Image
    
### sitk.MultiLabelSTAPLEImageFilter


    This filter performs a pixelwise combination of an arbitrary number of
    input images, where each of them represents a segmentation of the same
    scene (i.e., image).


    The labelings in the images are weighted relative to each other based
    on their "performance" as estimated by an expectation-maximization
    algorithm. In the process, a ground truth segmentation is estimated,
    and the estimated performances of the individual segmentations are
    relative to this estimated ground truth.

    The algorithm is based on the binary STAPLE algorithm by Warfield et
    al. as published originally in

    S. Warfield, K. Zou, W. Wells, "Validation of image segmentation and
    expert quality with an expectation-maximization algorithm" in MICCAI
    2002: Fifth International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer-Verlag,
    Heidelberg, Germany, 2002, pp. 298-306

    The multi-label algorithm implemented here is described in detail in

    T. Rohlfing, D. B. Russakoff, and C. R. Maurer, Jr., "Performance-
    based classifier combination in atlas-based image segmentation using
    expectation-maximization parameter estimation," IEEE Transactions on
    Medical Imaging, vol. 23, pp. 983-994, Aug. 2004.

    INPUTS
    All input volumes to this filter must be segmentations of an image,
    that is, they must have discrete pixel values where each value
    represents a different segmented object.
     Input volumes must all contain the same size RequestedRegions. Not all input images must contain all possible labels, but all label
    values must have the same meaning in all images.

    The filter can optionally be provided with estimates for the a priori
    class probabilities through the SetPriorProbabilities function. If no
    estimate is provided, one is automatically generated by analyzing the
    relative frequencies of the labels in the input images.

    OUTPUTS
    The filter produces a single output volume. Each output pixel contains
    the label that has the highest probability of being the correct label,
    based on the performance models of the individual segmentations. If
    the maximum probaility is not unique, i.e., if more than one label
    have a maximum probability, then an "undecided" label is assigned to
    that output pixel.
     By default, the label used for undecided pixels is the maximum label
    value used in the input images plus one. Since it is possible for an
    image with 8 bit pixel values to use all 256 possible label values, it
    is permissible to combine 8 bit (i.e., byte) images into a 16 bit
    (i.e., short) output image.

    In addition to the combined image, the estimated confusion matrices
    for each of the input segmentations can be obtained through the
    GetConfusionMatrix member function.

    PARAMETERS
    The label used for "undecided" labels can be set using
    SetLabelForUndecidedPixels. This functionality can be unset by calling
    UnsetLabelForUndecidedPixels.
     A termination threshold for the EM iteration can be defined by
    calling SetTerminationUpdateThreshold. The iteration terminates once
    no single parameter of any confusion matrix changes by less than this
    threshold. Alternatively, a maximum number of iterations can be
    specified by calling SetMaximumNumberOfIterations. The algorithm may
    still terminate after a smaller number of iterations if the
    termination threshold criterion is satisfied.

    EVENTS
    This filter invokes IterationEvent() at each iteration of the E-M
    algorithm. Setting the AbortGenerateData() flag will cause the
    algorithm to halt after the current iteration and produce results just
    as if it had converged. The algorithm makes no attempt to report its
    progress since the number of iterations needed cannot be known in
    advance.

    Torsten Rohlfing, SRI International, Neuroscience Program

    See:
     itk::simple::MultiLabelSTAPLE for the procedural interface


    C++ includes: sitkMultiLabelSTAPLEImageFilter.h

    
### sitk.Multiply
    Multiply(Image image1, Image image2) -> Image
    Multiply(Image image1, double constant) -> Image
    Multiply(double constant, Image image2) -> Image



    
### sitk.MultiplyImageFilter


    Pixel-wise multiplication of two images.


    This class is templated over the types of the two input images and the
    type of the output image. Numeric conversions (castings) are done by
    the C++ defaults.
    See:
     itk::simple::Multiply for the procedural interface

     itk::MultiplyImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkMultiplyImageFilter.h

    
### sitk.N4BiasFieldCorrection
    N4BiasFieldCorrection(Image image, Image maskImage, double convergenceThreshold=0.001, VectorUInt32 maximumNumberOfIterations, double biasFieldFullWidthAtHalfMaximum=0.15, double wienerFilterNoise=0.01, uint32_t numberOfHistogramBins=200, VectorUInt32 numberOfControlPoints, uint32_t splineOrder=3, bool useMaskLabel=True, uint8_t maskLabel=1) -> Image
    N4BiasFieldCorrection(Image image, double convergenceThreshold=0.001, VectorUInt32 maximumNumberOfIterations, double biasFieldFullWidthAtHalfMaximum=0.15, double wienerFilterNoise=0.01, uint32_t numberOfHistogramBins=200, VectorUInt32 numberOfControlPoints, uint32_t splineOrder=3, bool useMaskLabel=True, uint8_t maskLabel=1) -> Image



    
### sitk.N4BiasFieldCorrectionImageFilter


    Implementation of the N4 bias field correction algorithm.


    The nonparametric nonuniform intensity normalization (N3) algorithm,
    as introduced by Sled et al. in 1998 is a method for correcting
    nonuniformity associated with MR images. The algorithm assumes a
    simple parametric model (Gaussian) for the bias field and does not
    require tissue class segmentation. In addition, there are only a
    couple of parameters to tune with the default values performing quite
    well. N3 has been publicly available as a set of perl scripts ( http://www.bic.mni.mcgill.ca/ServicesSoftwareAdvancedImageProcessingTo
    ols/HomePage )

    The N4 algorithm, encapsulated with this class, is a variation of the
    original N3 algorithm with the additional benefits of an improved
    B-spline fitting routine which allows for multiple resolutions to be
    used during the correction process. We also modify the iterative
    update component of algorithm such that the residual bias field is
    continually updated

    Notes for the user:
    Since much of the image manipulation is done in the log space of the
    intensities, input images with negative and small values (< 1) can
    produce poor results.

    The original authors recommend performing the bias field correction on
    a downsampled version of the original image.

    A binary mask or a weighted image can be supplied. If a binary mask is
    specified, those voxels in the input image which correspond to the
    voxels in the mask image are used to estimate the bias field. If a
    UseMaskLabel value is set to true, only voxels in the MaskImage that
    match the MaskLabel will be used; otherwise, all non-zero voxels in
    the MaskImage will be masked. If a confidence image is specified, the
    input voxels are weighted in the b-spline fitting routine according to
    the confidence voxel values.

    The filter returns the corrected image. If the bias field is wanted,
    one can reconstruct it using the class
    itkBSplineControlPointImageFilter. See the IJ article and the test
    file for an example.

    The 'Z' parameter in Sled's 1998 paper is the square root of the class
    variable 'm_WienerFilterNoise'.
     The basic algorithm iterates between sharpening the intensity
    histogram of the corrected input image and spatially smoothing those
    results with a B-spline scalar field estimate of the bias field.


    Nicholas J. Tustison
     Contributed by Nicholas J. Tustison, James C. Gee in the Insight
    Journal paper: https://hdl.handle.net/10380/3053

    REFERENCE
     J.G. Sled, A.P. Zijdenbos and A.C. Evans. "A Nonparametric Method
    for Automatic Correction of Intensity Nonuniformity in Data" IEEE
    Transactions on Medical Imaging, Vol 17, No 1. Feb 1998.

    N.J. Tustison, B.B. Avants, P.A. Cook, Y. Zheng, A. Egan, P.A.
    Yushkevich, and J.C. Gee. "N4ITK: Improved N3 Bias Correction" IEEE
    Transactions on Medical Imaging, 29(6):1310-1320, June 2010.
    See:
     itk::simple::N4BiasFieldCorrection for the procedural interface

     itk::N4BiasFieldCorrectionImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkN4BiasFieldCorrectionImageFilter.h

    
### sitk.NaryAdd
    NaryAdd(VectorOfImage images) -> Image
    NaryAdd(Image image1) -> Image
    NaryAdd(Image image1, Image image2) -> Image
    NaryAdd(Image image1, Image image2, Image image3) -> Image
    NaryAdd(Image image1, Image image2, Image image3, Image image4) -> Image
    NaryAdd(Image image1, Image image2, Image image3, Image image4, Image image5) -> Image
    
### sitk.NaryAddImageFilter


    Pixel-wise addition of N images.


    This class is templated over the types of the input images and the
    type of the output image. Numeric conversions (castings) are done by
    the C++ defaults.

    The pixel type of the input images must have a valid definition of the
    operator+ with each other. This condition is required because
    internally this filter will perform the operation


    Additionally the type resulting from the sum, will be cast to the
    pixel type of the output image.

    The total operation over one pixel will be


    For example, this filter could be used directly for adding images
    whose pixels are vectors of the same dimension, and to store the
    resulting vector in an output image of vector pixels.


    WARNING:
    No numeric overflow checking is performed in this filter.

    See:
     itk::simple::NaryAdd for the procedural interface


    C++ includes: sitkNaryAddImageFilter.h

    
### sitk.NaryMaximum
    NaryMaximum(VectorOfImage images) -> Image
    NaryMaximum(Image image1) -> Image
    NaryMaximum(Image image1, Image image2) -> Image
    NaryMaximum(Image image1, Image image2, Image image3) -> Image
    NaryMaximum(Image image1, Image image2, Image image3, Image image4) -> Image
    NaryMaximum(Image image1, Image image2, Image image3, Image image4, Image image5) -> Image
    
### sitk.NaryMaximumImageFilter


    Computes the pixel-wise maximum of several images.


    This class is templated over the types of the input images and the
    type of the output image. Numeric conversions (castings) are done by
    the C++ defaults.

    The pixel type of the output images must have a valid definition of
    the operator<. This condition is required because internally this
    filter will perform an operation similar to:

     (where current_maximum is also of type OutputPixelType)

    for each of the n input images.

    For example, this filter could be used directly to find a "maximum
    projection" of a series of images, often used in preliminary analysis
    of time-series data.


    Zachary Pincus
     This filter was contributed by Zachary Pincus from the Department of
    Biochemistry and Program in Biomedical Informatics at Stanford
    University School of Medicine


    See:
     itk::simple::NaryMaximum for the procedural interface


    C++ includes: sitkNaryMaximumImageFilter.h

    
### sitk.NeighborhoodConnected
    NeighborhoodConnected(Image image1, VectorUIntList seedList, double lower=0, double upper=1, VectorUInt32 radius, double replaceValue=1) -> Image



    itk::simple::NeighborhoodConnectedImageFilter Functional Interface

    This function directly calls the execute method of NeighborhoodConnectedImageFilter in order to support a fully functional API


    
### sitk.NeighborhoodConnectedImageFilter


    Label pixels that are connected to a seed and lie within a
    neighborhood.


    NeighborhoodConnectedImageFilter labels pixels with ReplaceValue that are connected to an initial Seed
    AND whose neighbors all lie within a Lower and Upper threshold range.
    See:
     itk::simple::NeighborhoodConnected for the procedural interface

     itk::NeighborhoodConnectedImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkNeighborhoodConnectedImageFilter.h

    
### sitk.Noise
    Noise(Image image1, VectorUInt32 radius) -> Image



    Calculate the local noise in an image.


    This function directly calls the execute method of NoiseImageFilter in order to support a procedural API


    See:
     itk::simple::NoiseImageFilter for the object oriented interface



    
### sitk.NoiseImageFilter


    Calculate the local noise in an image.


    Computes an image where a given pixel is the standard deviation of the
    pixels in a neighborhood about the corresponding input pixel. This
    serves as an estimate of the local noise (or texture) in an image.
    Currently, this noise estimate assume a piecewise constant image. This
    filter should be extended to fitting a (hyper) plane to the
    neighborhood and calculating the standard deviation of the residuals
    to this (hyper) plane.


    See:
     Image

     Neighborhood

     NeighborhoodOperator

     NeighborhoodIterator

     itk::simple::Noise for the procedural interface

     itk::NoiseImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkNoiseImageFilter.h

    
### sitk.Normalize
    Normalize(Image image1) -> Image



    Normalize an image by setting its mean to zero and variance to one.


    This function directly calls the execute method of NormalizeImageFilter in order to support a procedural API


    See:
     itk::simple::NormalizeImageFilter for the object oriented interface



    
### sitk.NormalizeImageFilter


    Normalize an image by setting its mean to zero and variance to one.


    NormalizeImageFilter shifts and scales an image so that the pixels in the image have a
    zero mean and unit variance. This filter uses StatisticsImageFilter to compute the mean and variance of the input and then applies ShiftScaleImageFilter to shift and scale the pixels.

    NB: since this filter normalizes the data to lie within -1 to 1,
    integral types will produce an image that DOES NOT HAVE a unit
    variance.


    See:
     NormalizeToConstantImageFilter

     itk::simple::Normalize for the procedural interface

     itk::NormalizeImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkNormalizeImageFilter.h

    
### sitk.NormalizeToConstant
    NormalizeToConstant(Image image1, double constant=1.0) -> Image



    Scales image pixel intensities to make the sum of all pixels equal a
    user-defined constant.


    This function directly calls the execute method of NormalizeToConstantImageFilter in order to support a procedural API


    See:
     itk::simple::NormalizeToConstantImageFilter for the object oriented interface



    
### sitk.NormalizeToConstantImageFilter


    Scales image pixel intensities to make the sum of all pixels equal a
    user-defined constant.


    The default value of the constant is 1. It can be changed with SetConstant() .

    This transform is especially useful for normalizing a convolution
    kernel.

    This code was contributed in the Insight Journal paper: "FFT based
    convolution" by Lehmann G. https://hdl.handle.net/10380/3154


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.

    See:
     NormalizeImageFilter

     StatisticsImageFilter

     DivideImageFilter

     itk::simple::NormalizeToConstant for the procedural interface

     itk::NormalizeToConstantImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkNormalizeToConstantImageFilter.h

    
### sitk.NormalizedCorrelation
    NormalizedCorrelation(Image image, Image maskImage, Image templateImage) -> Image



    Computes the normalized correlation of an image and a template.


    This function directly calls the execute method of NormalizedCorrelationImageFilter in order to support a procedural API


    See:
     itk::simple::NormalizedCorrelationImageFilter for the object oriented interface



    
### sitk.NormalizedCorrelationImageFilter


    Computes the normalized correlation of an image and a template.


    This filter calculates the normalized correlation between an image and
    the template. Normalized correlation is frequently use in feature
    detection because it is invariant to local changes in contrast.

    The filter can be given a mask. When presented with an input image and
    a mask, the normalized correlation is only calculated at those pixels
    under the mask.


    See:
     Image

     Neighborhood

     NeighborhoodOperator

     NeighborhoodIterator

     itk::simple::NormalizedCorrelation for the procedural interface

     itk::NormalizedCorrelationImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkNormalizedCorrelationImageFilter.h

    
### sitk.Not
    Not(Image image1) -> Image



    Implements the NOT logical operator pixel-wise on an image.


    This function directly calls the execute method of NotImageFilter in order to support a procedural API


    See:
     itk::simple::NotImageFilter for the object oriented interface



    
### sitk.NotEqual
    NotEqual(Image image1, Image image2, uint8_t backgroundValue=0, uint8_t foregroundValue=1) -> Image
    NotEqual(Image image1, double constant, uint8_t backgroundValue=0, uint8_t foregroundValue=1) -> Image
    NotEqual(double constant, Image image2, uint8_t backgroundValue=0, uint8_t foregroundValue=1) -> Image



    
### sitk.NotEqualImageFilter


    Implements pixel-wise generic operation of two images, or of an image
    and a constant.


    This class is parameterized over the types of the two input images and
    the type of the output image. It is also parameterized by the
    operation to be applied. A Functor style is used.

    The constant must be of the same type than the pixel type of the
    corresponding image. It is wrapped in a SimpleDataObjectDecorator so it can be updated through the pipeline. The SetConstant() and
    GetConstant() methods are provided as shortcuts to set or get the
    constant value without manipulating the decorator.


    See:
     UnaryFunctorImageFilter TernaryFunctorImageFilter

     itk::simple::NotEqual for the procedural interface

     itk::BinaryFunctorImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkNotEqualImageFilter.h

    
### sitk.NotImageFilter


    Implements the NOT logical operator pixel-wise on an image.


    This class is templated over the type of an input image and the type
    of the output image. Numeric conversions (castings) are done by the
    C++ defaults.

    Since the logical NOT operation operates only on boolean types, the
    input type must be implicitly convertible to bool, which is only
    defined in C++ for integer types, the images passed to this filter
    must comply with the requirement of using integer pixel type.

    The total operation over one pixel will be


    Where "!" is the unary Logical NOT operator in C++.
    See:
     itk::simple::Not for the procedural interface

     itk::NotImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkNotImageFilter.h

    
### sitk.ObjectnessMeasure
    ObjectnessMeasure(Image image1, double alpha=0.5, double beta=0.5, double gamma=5, bool scaleObjectnessMeasure=True, unsigned int objectDimension=1, bool brightObject=True) -> Image



    Enhance M-dimensional objects in N-dimensional images.


    This function directly calls the execute method of ObjectnessMeasureImageFilter in order to support a procedural API


    See:
     itk::simple::ObjectnessMeasureImageFilter for the object oriented interface



    
### sitk.ObjectnessMeasureImageFilter


    Enhance M-dimensional objects in N-dimensional images.


    This filter is a generalization of Frangi's vesselness measurement for
    detecting M-dimensional object in N-dimensional space. For example a
    vessel is a 1-D object in 3-D space. The filter can enhance blob-like
    structures (M=0), vessel-like structures (M=1), 2D plate-like
    structures (M=2), hyper-plate-like structures (M=3) in N-dimensional
    images, with M<N.

    This filter takes a scalar image as input and produces a real valued
    image as output which contains the objectness measure at each pixel.
    Internally, it computes a Hessian via discrete central differences.
    Before applying this filter it is expected that a Gaussian smoothing
    filter at an appropriate scale (sigma) was applied to the input image.

    The enhancement is based on the eigenvalues of the Hessian matrix. For
    the Frangi's vesselness case were M=1 and N=3 we have the 3
    eigenvalues such that $ | \lambda_1 | < | \lambda_2 | < |\lambda_3 | $ . The formula follows:

    \[ R_A = \frac{|\lambda_2|}{|\lambda_3|}, \; R_B =
    \frac{|\lambda_2|}{|\lambda_2\lambda_3|}, \; S =
    \sqrt{\lambda_1^2+\lambda_2^2+\lambda_3^2} \] \[ V_{\sigma}= \begin{cases}
    (1-e^{-\frac{R_A^2}{2\alpha^2}}) \cdot
    e^{\frac{R_B^2}{2\beta^2}} \cdot
    (1-e^{-\frac{S^2}{2\gamma^2}}) & \text{if } \lambda_2<0
    \text{ and } \lambda_3<0 \text{,}\\ 0 &
    \text{otherwise} \end{cases} \]

    References
    Antiga, L. Generalizing vesselness with respect to dimensionality and
    shape. https://hdl.handle.net/1926/576

    Frangi, AF, Niessen, WJ, Vincken, KL, & Viergever, MA (1998).
    Multiscale Vessel Enhancement Filtering. In Wells, WM, Colchester, A,
    & Delp, S, Editors, MICCAI '98 Medical Image Computing and Computer-Assisted Intervention, Lecture Notes in
    Computer Science, pages 130-137, Springer Verlag, 1998.

    See:
     itk::HessianToObjectnessMeasureImageFilter

     itk::simple::ObjectnessMeasure for the procedural interface

     itk::ObjectnessMeasureImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkObjectnessMeasureImageFilter.h

    
### sitk.OpeningByReconstruction
    OpeningByReconstruction(Image arg1, uint32_t radius=1, itk::simple::KernelEnum kernel, bool fullyConnected=False, bool preserveIntensities=False) -> Image
    OpeningByReconstruction(Image arg1, VectorUInt32 vectorRadius, itk::simple::KernelEnum kernel, bool fullyConnected=False, bool preserveIntensities=False) -> Image



    itk::simple::OpeningByReconstructionImageFilter Functional Interface

    This function directly calls the execute method of OpeningByReconstructionImageFilter in order to support a fully functional API


    
### sitk.OpeningByReconstructionImageFilter


    Opening by reconstruction of an image.


    This filter preserves regions, in the foreground, that can completely
    contain the structuring element. At the same time, this filter
    eliminates all other regions of foreground pixels. Contrary to the
    mophological opening, the opening by reconstruction preserves the
    shape of the components that are not removed by erosion. The opening
    by reconstruction of an image "f" is defined as:

    OpeningByReconstruction(f) = DilationByRecontruction(f, Erosion(f)).

    Opening by reconstruction not only removes structures destroyed by the
    erosion, but also levels down the contrast of the brightest regions.
    If PreserveIntensities is on, a subsequent reconstruction by dilation
    using a marker image that is the original image for all unaffected
    pixels.

    Opening by reconstruction is described in Chapter 6.3.9 of Pierre
    Soille's book "Morphological Image Analysis: Principles and
    Applications", Second Edition, Springer, 2003.


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.

    See:
     GrayscaleMorphologicalOpeningImageFilter

     itk::simple::OpeningByReconstruction for the procedural interface

     itk::OpeningByReconstructionImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkOpeningByReconstructionImageFilter.h

    
### sitk.Or
    Or(Image image1, Image image2) -> Image
    Or(Image image1, int constant) -> Image
    Or(int constant, Image image2) -> Image



    
### sitk.OrImageFilter


    Implements the OR bitwise operator pixel-wise between two images.


    This class is templated over the types of the two input images and the
    type of the output image. Numeric conversions (castings) are done by
    the C++ defaults.

    Since the bitwise OR operation is only defined in C++ for integer
    types, the images passed to this filter must comply with the
    requirement of using integer pixel type.

    The total operation over one pixel will be


    Where "|" is the boolean OR operator in C++.
    See:
     itk::simple::Or for the procedural interface

     itk::OrImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkOrImageFilter.h

    
### sitk.OtsuMultipleThresholds
    OtsuMultipleThresholds(Image image1, uint8_t numberOfThresholds=1, uint8_t labelOffset=0, uint32_t numberOfHistogramBins=128, bool valleyEmphasis=False) -> Image



    Threshold an image using multiple Otsu Thresholds.


    This function directly calls the execute method of OtsuMultipleThresholdsImageFilter in order to support a procedural API


    See:
     itk::simple::OtsuMultipleThresholdsImageFilter for the object oriented interface



    
### sitk.OtsuMultipleThresholdsImageFilter


    Threshold an image using multiple Otsu Thresholds.


    This filter creates a labeled image that separates the input image
    into various classes. The filter computes the thresholds using the OtsuMultipleThresholdsCalculator and applies those thresholds to the input image using the ThresholdLabelerImageFilter . The NumberOfHistogramBins and NumberOfThresholds can be set for the
    Calculator. The LabelOffset can be set for the ThresholdLabelerImageFilter .

    This filter also includes an option to use the valley emphasis
    algorithm from H.F. Ng, "Automatic thresholding for defect
    detection", Pattern Recognition Letters, (27): 1644-1649, 2006. The
    valley emphasis algorithm is particularly effective when the object to
    be thresholded is small. See the following tests for examples:
    itkOtsuMultipleThresholdsImageFilterTest3 and
    itkOtsuMultipleThresholdsImageFilterTest4 To use this algorithm,
    simple call the setter: SetValleyEmphasis(true) It is turned off by
    default.


    See:
     ScalarImageToHistogramGenerator

     OtsuMultipleThresholdsCalculator

     ThresholdLabelerImageFilter

     itk::simple::OtsuMultipleThresholds for the procedural interface

     itk::OtsuMultipleThresholdsImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkOtsuMultipleThresholdsImageFilter.h

    
### sitk.OtsuThreshold
    OtsuThreshold(Image image, Image maskImage, uint8_t insideValue=1, uint8_t outsideValue=0, uint32_t numberOfHistogramBins=128, bool maskOutput=True, uint8_t maskValue=255) -> Image
    OtsuThreshold(Image image, uint8_t insideValue=1, uint8_t outsideValue=0, uint32_t numberOfHistogramBins=128, bool maskOutput=True, uint8_t maskValue=255) -> Image



    
### sitk.OtsuThresholdImageFilter


    Threshold an image using the Otsu Threshold.


    This filter creates a binary thresholded image that separates an image
    into foreground and background components. The filter computes the
    threshold using the OtsuThresholdCalculator and applies that threshold to the input image using the BinaryThresholdImageFilter .


    Richard Beare

    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.
     This implementation was taken from the Insight Journal paper: https://hdl.handle.net/10380/3279 or http://www.insight-journal.org/browse/publication/811


    See:
     HistogramThresholdImageFilter

     itk::simple::OtsuThreshold for the procedural interface

     itk::OtsuThresholdImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkOtsuThresholdImageFilter.h

    
### sitk.Paste
    Paste(Image destinationImage, Image sourceImage, VectorUInt32 sourceSize, VectorInt32 sourceIndex, VectorInt32 destinationIndex) -> Image



    Paste an image into another image.


    This function directly calls the execute method of PasteImageFilter in order to support a procedural API


    See:
     itk::simple::PasteImageFilter for the object oriented interface



    
### sitk.PasteImageFilter


    Paste an image into another image.


    PasteImageFilter allows you to take a section of one image and paste into another
    image. The SetDestinationIndex() method prescribes where in the first input to start pasting data from
    the second input. The SetSourceRegion method prescribes the section of
    the second image to paste into the first. If the output requested
    region does not include the SourceRegion after it has been
    repositioned to DestinationIndex, then the output will just be a copy
    of the input.

    The two inputs and output image will have the same pixel type.
    See:
     itk::simple::Paste for the procedural interface

     itk::PasteImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkPasteImageFilter.h

    
### sitk.PatchBasedDenoising
    PatchBasedDenoising(Image image1, itk::simple::PatchBasedDenoisingImageFilter::NoiseModelType noiseModel, double kernelBandwidthSigma=400.0, uint32_t patchRadius=4, uint32_t numberOfIterations=1, uint32_t numberOfSamplePatches=200, double sampleVariance=400.0, double noiseSigma=0.0, double noiseModelFidelityWeight=0.0) -> Image
    PatchBasedDenoising(Image image1, double kernelBandwidthSigma=400.0, uint32_t patchRadius=4, uint32_t numberOfIterations=1, uint32_t numberOfSamplePatches=200, double sampleVariance=400.0) -> Image



    
### sitk.PatchBasedDenoisingImageFilter


    Derived class implementing a specific patch-based denoising algorithm,
    as detailed below.


    This class is derived from the base class PatchBasedDenoisingBaseImageFilter ; please refer to the documentation of the base class first. This
    class implements a denoising filter that uses iterative non-local, or
    semi-local, weighted averaging of image patches for image denoising.
    The intensity at each pixel 'p' gets updated as a weighted average of
    intensities of a chosen subset of pixels from the image.

    This class implements the denoising algorithm using a Gaussian kernel
    function for nonparametric density estimation. The class implements a
    scheme to automatically estimated the kernel bandwidth parameter
    (namely, sigma) using leave-one-out cross validation. It implements
    schemes for random sampling of patches non-locally (from the entire
    image) as well as semi-locally (from the spatial proximity of the
    pixel being denoised at the specific point in time). It implements a
    specific scheme for defining patch weights (mask) as described in
    Awate and Whitaker 2005 IEEE CVPR and 2006 IEEE TPAMI.


    See:
     PatchBasedDenoisingBaseImageFilter

     itk::PatchBasedDenoisingImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkPatchBasedDenoisingImageFilter.h

    
### sitk.PermuteAxes
    PermuteAxes(Image image1, VectorUInt32 order) -> Image



    Permutes the image axes according to a user specified order.


    This function directly calls the execute method of PermuteAxesImageFilter in order to support a procedural API


    See:
     itk::simple::PermuteAxesImageFilter for the object oriented interface



    
### sitk.PermuteAxesImageFilter


    Permutes the image axes according to a user specified order.


    PermuateAxesImageFilter permutes the image axes according to a user
    specified order. The permutation order is set via method SetOrder(
    order ) where the input is an array of ImageDimension number of
    unsigned int. The elements of the array must be a rearrangment of the
    numbers from 0 to ImageDimension - 1.

    The i-th axis of the output image corresponds with the order[i]-th
    axis of the input image.

    The output meta image information (LargestPossibleRegion, spacing,
    origin) is computed by permuting the corresponding input meta
    information.
    See:
     itk::simple::PermuteAxes for the procedural interface

     itk::PermuteAxesImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkPermuteAxesImageFilter.h

    
### sitk.PhysicalPointImageSource


    Generate an image of the physical locations of each pixel.


    This image source supports image which have a multi-component pixel
    equal to the image dimension, and variable length VectorImages. It is
    recommended that the component type be a real valued type.
    See:
     itk::simple::PhysicalPointSource for the procedural interface

     itk::PhysicalPointImageSource for the Doxygen on the original ITK class.


    C++ includes: sitkPhysicalPointImageSource.h

    
### sitk.PhysicalPointSource
    PhysicalPointSource(itk::simple::PixelIDValueEnum outputPixelType, VectorUInt32 size, VectorDouble origin, VectorDouble spacing, VectorDouble direction) -> Image



    Generate an image of the physical locations of each pixel.


    This function directly calls the execute method of PhysicalPointImageSource in order to support a procedural API


    See:
     itk::simple::PhysicalPointImageSource for the object oriented interface



    
### sitk.Pow
    Pow(Image image1, Image image2) -> Image
    Pow(Image image1, double constant) -> Image
    Pow(double constant, Image image2) -> Image



    
### sitk.PowImageFilter


    Computes the powers of 2 images.


    This class is templated over the types of the two input images and the
    type of the output image. Numeric conversions (castings) are done by
    the C++ defaults.

    The output of the pow function will be cast to the pixel type of the
    output image.

    The total operation over one pixel will be

    The pow function can be applied to two images with the following:

    Additionally, this filter can be used to raise every pixel of an image
    to a power of a constant by using
    See:
     itk::simple::Pow for the procedural interface

     itk::PowImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkPowImageFilter.h

    
### sitk.ProcessObject


    Base class for SimpleITK classes based on ProcessObject.

    C++ includes: sitkProcessObject.h

    
### sitk.ProcessObject_GetGlobalDefaultCoordinateToleranceProcessObject_GetGlobalDefaultCoordinateTolerance() -> double
### sitk.ProcessObject_GetGlobalDefaultDebugProcessObject_GetGlobalDefaultDebug() -> bool
### sitk.ProcessObject_GetGlobalDefaultDirectionToleranceProcessObject_GetGlobalDefaultDirectionTolerance() -> double
### sitk.ProcessObject_GetGlobalDefaultNumberOfThreadsProcessObject_GetGlobalDefaultNumberOfThreads() -> unsigned int
### sitk.ProcessObject_GetGlobalWarningDisplayProcessObject_GetGlobalWarningDisplay() -> bool
### sitk.ProcessObject_GlobalDefaultDebugOffProcessObject_GlobalDefaultDebugOff()
### sitk.ProcessObject_GlobalDefaultDebugOnProcessObject_GlobalDefaultDebugOn()
### sitk.ProcessObject_GlobalWarningDisplayOffProcessObject_GlobalWarningDisplayOff()
### sitk.ProcessObject_GlobalWarningDisplayOnProcessObject_GlobalWarningDisplayOn()
### sitk.ProcessObject_SetGlobalDefaultCoordinateToleranceProcessObject_SetGlobalDefaultCoordinateTolerance(double arg2)
### sitk.ProcessObject_SetGlobalDefaultDebugProcessObject_SetGlobalDefaultDebug(bool debugFlag)
### sitk.ProcessObject_SetGlobalDefaultDirectionToleranceProcessObject_SetGlobalDefaultDirectionTolerance(double arg2)
### sitk.ProcessObject_SetGlobalDefaultNumberOfThreadsProcessObject_SetGlobalDefaultNumberOfThreads(unsigned int n)
### sitk.ProcessObject_SetGlobalWarningDisplayProcessObject_SetGlobalWarningDisplay(bool flag)
### sitk.ProjectedLandweberDeconvolution
    ProjectedLandweberDeconvolution(Image image1, Image image2, double alpha=0.1, int numberOfIterations=1, bool normalize=False, itk::simple::ProjectedLandweberDeconvolutionImageFilter::BoundaryConditionType boundaryCondition, itk::simple::ProjectedLandweberDeconvolutionImageFilter::OutputRegionModeType outputRegionMode) -> Image



    Deconvolve an image using the projected Landweber deconvolution
    algorithm.


    This function directly calls the execute method of ProjectedLandweberDeconvolutionImageFilter in order to support a procedural API


    See:
     itk::simple::ProjectedLandweberDeconvolutionImageFilter for the object oriented interface



    
### sitk.ProjectedLandweberDeconvolutionImageFilter


    Deconvolve an image using the projected Landweber deconvolution
    algorithm.


    This filter performs the same calculation per iteration as the LandweberDeconvolutionImageFilter . However, at each iteration, negative pixels in the intermediate
    result are projected (set) to zero. This is useful if the solution is
    assumed to always be non-negative, which is the case when dealing with
    images formed by counting photons, for example.

    This code was adapted from the Insight Journal contribution:

    "Deconvolution: infrastructure and reference algorithms" by Gaetan
    Lehmann https://hdl.handle.net/10380/3207


    Gaetan Lehmann, Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France

    Cory Quammen, The University of North Carolina at Chapel Hill

    See:
     IterativeDeconvolutionImageFilter

     RichardsonLucyDeconvolutionImageFilter

     LandweberDeconvolutionImageFilter

     itk::simple::ProjectedLandweberDeconvolution for the procedural interface

     itk::ProjectedLandweberDeconvolutionImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkProjectedLandweberDeconvolutionImageFilter.h

    
### sitk.PyCommandProxy of C++ itk::simple::PyCommand class.
### sitk.Rank
    Rank(Image image1, double rank=0.5, VectorUInt32 radius) -> Image



    Rank filter of a greyscale image.


    This function directly calls the execute method of RankImageFilter in order to support a procedural API


    See:
     itk::simple::RankImageFilter for the object oriented interface



    
### sitk.RankImageFilter


    Rank filter of a greyscale image.


    Nonlinear filter in which each output pixel is a user defined rank of
    input pixels in a user defined neighborhood. The default rank is 0.5
    (median). The boundary conditions are different to the standard
    itkMedianImageFilter. In this filter the neighborhood is cropped at
    the boundary, and is therefore smaller.

    This filter uses a recursive implementation - essentially the one by
    Huang 1979, I believe, to compute the rank, and is therefore usually a
    lot faster than the direct implementation. The extensions to Huang are
    support for arbitrary pixel types (using c++ maps) and arbitrary
    neighborhoods. I presume that these are not new ideas.

    This filter is based on the sliding window code from the
    consolidatedMorphology package on InsightJournal.

    The structuring element is assumed to be composed of binary values
    (zero or one). Only elements of the structuring element having values
    > 0 are candidates for affecting the center pixel.

    This code was contributed in the Insight Journal paper: "Efficient
    implementation of kernel filtering" by Beare R., Lehmann G https://hdl.handle.net/1926/555 http://www.insight-journal.org/browse/publication/160


    See:
     MedianImageFilter

    Richard Beare

    See:
     itk::simple::Rank for the procedural interface

     itk::RankImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkRankImageFilter.h

    
### sitk.ReadImage
    ReadImage(VectorString fileNames, itk::simple::PixelIDValueEnum outputPixelType, std::string const & imageIO) -> Image
    ReadImage(std::string const & filename, itk::simple::PixelIDValueEnum outputPixelType, std::string const & imageIO) -> Image



    ReadImage is a procedural interface to the ImageFileReader class which is convenient for most image reading tasks.




    Parameters:

    filename:
    the filename of an Image e.g. "cthead.mha"

    outputPixelType:
    see ImageReaderBase::SetOutputPixelType

    imageIO:
    see ImageReaderBase::SetImageIO


    See:
     itk::simple::ImageFileReader for reading a single file.

     itk::simple::ImageSeriesReader for reading a series and meta-data dictionaries.



    
### sitk.ReadTransform
    ReadTransform(std::string const & filename) -> Transform



    
### sitk.RealAndImaginaryToComplex
    RealAndImaginaryToComplex(Image image1, Image image2) -> Image



    ComposeImageFilter combine several scalar images into a multicomponent image.


    This function directly calls the execute method of RealAndImaginaryToComplexImageFilter in order to support a procedural API


    See:
     itk::simple::RealAndImaginaryToComplexImageFilter for the object oriented interface



    
### sitk.RealAndImaginaryToComplexImageFilter


    ComposeImageFilter combine several scalar images into a multicomponent image.


    ComposeImageFilter combine several scalar images into an itk::Image of vector pixel ( itk::Vector , itk::RGBPixel , ...), of std::complex pixel, or in an itk::VectorImage .

    Inputs and Usage
     All input images are expected to have the same template parameters
    and have the same size and origin.

    See:
     VectorImage

     VectorIndexSelectionCastImageFilter

     itk::simple::RealAndImaginaryToComplex for the procedural interface

     itk::ComposeImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkRealAndImaginaryToComplexImageFilter.h

    
### sitk.RealToHalfHermitianForwardFFT
    RealToHalfHermitianForwardFFT(Image image1) -> Image



    Base class for specialized real-to-complex forward Fast Fourier Transform .


    This function directly calls the execute method of RealToHalfHermitianForwardFFTImageFilter in order to support a procedural API


    See:
     itk::simple::RealToHalfHermitianForwardFFTImageFilter for the object oriented interface



    
### sitk.RealToHalfHermitianForwardFFTImageFilter


    Base class for specialized real-to-complex forward Fast Fourier Transform .


    This is a base class for the "forward" or "direct" discrete
    Fourier Transform . This is an abstract base class: the actual implementation is
    provided by the best child class available on the system when the
    object is created via the object factory system.

    This class transforms a real input image into its complex Fourier
    transform. The Fourier transform of a real input image has Hermitian
    symmetry: $ f(\mathbf{x}) = f^*(-\mathbf{x}) $ . That is, when the result of the transform is split in half along
    the X-dimension, the values in the second half of the transform are
    the complex conjugates of values in the first half reflected about the
    center of the image in each dimension. This filter takes advantage of
    the Hermitian symmetry property and reduces the size of the output in
    the first dimension to N/2+1, where N is the size of the input image
    in that dimension and the division by 2 is rounded down.


    See:
     HalfHermitianToRealInverseFFTImageFilter

     ForwardFFTImageFilter

     itk::simple::RealToHalfHermitianForwardFFT for the procedural interface

     itk::RealToHalfHermitianForwardFFTImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkRealToHalfHermitianForwardFFTImageFilter.h

    
### sitk.ReconstructionByDilation
    ReconstructionByDilation(Image image1, Image image2, bool fullyConnected=False, bool useInternalCopy=True) -> Image



    grayscale reconstruction by dilation of an image


    This function directly calls the execute method of ReconstructionByDilationImageFilter in order to support a procedural API


    See:
     itk::simple::ReconstructionByDilationImageFilter for the object oriented interface



    
### sitk.ReconstructionByDilationImageFilter


    grayscale reconstruction by dilation of an image


    Reconstruction by dilation operates on a "marker" image and a
    "mask" image, and is defined as the dilation of the marker image
    with respect to the mask image iterated until stability.

    The marker image must be less than or equal to the mask image (on a
    pixel by pixel basis).

    Geodesic morphology is described in Chapter 6.2 of Pierre Soille's
    book "Morphological Image Analysis: Principles and Applications",
    Second Edition, Springer, 2003.

    Algorithm implemented in this filter is based on algorithm described
    by Kevin Robinson and Paul F. Whelan in "Efficient Morphological
    Reconstruction: A Downhill Filter", Pattern Recognition Letters,
    Volume 25, Issue 15, November 2004, Pages 1759-1767.

    The algorithm, a description of the transform and some applications
    can be found in "Morphological Grayscale Reconstruction in Image
    Analysis:  Applications and Efficient Algorithms", Luc Vincent, IEEE
    Transactions on image processing, Vol. 2, April 1993.


    Richard Beare. Department of Medicine, Monash University, Melbourne,
    Australia.

    See:
     MorphologyImageFilter , GrayscaleDilateImageFilter , GrayscaleFunctionDilateImageFilter , BinaryDilateImageFilter , ReconstructionByErosionImageFilter , OpeningByReconstructionImageFilter , ClosingByReconstructionImageFilter , ReconstructionImageFilter

     itk::simple::ReconstructionByDilation for the procedural interface

     itk::ReconstructionByDilationImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkReconstructionByDilationImageFilter.h

    
### sitk.ReconstructionByErosion
    ReconstructionByErosion(Image image1, Image image2, bool fullyConnected=False, bool useInternalCopy=True) -> Image



    grayscale reconstruction by erosion of an image


    This function directly calls the execute method of ReconstructionByErosionImageFilter in order to support a procedural API


    See:
     itk::simple::ReconstructionByErosionImageFilter for the object oriented interface



    
### sitk.ReconstructionByErosionImageFilter


    grayscale reconstruction by erosion of an image


    Reconstruction by erosion operates on a "marker" image and a
    "mask" image, and is defined as the erosion of the marker image with
    respect to the mask image iterated until stability.

    The marker image must be less than or equal to the mask image (on a
    pixel by pixel basis).

    Geodesic morphology is described in Chapter 6.2 of Pierre Soille's
    book "Morphological Image Analysis: Principles and Applications",
    Second Edition, Springer, 2003.

    Algorithm implemented in this filter is based on algorithm described
    by Kevin Robinson and Paul F. Whelan in "Efficient Morphological
    Reconstruction: A Downhill Filter", Pattern Recognition Letters,
    Volume 25, Issue 15, November 2004, Pages 1759-1767.

    The algorithm, a description of the transform and some applications
    can be found in "Morphological Grayscale Reconstruction in Image
    Analysis:  Applications and Efficient Algorithms", Luc Vincent, IEEE
    Transactions on image processing, Vol. 2, April 1993.


    Richard Beare. Department of Medicine, Monash University, Melbourne,
    Australia.

    See:
     MorphologyImageFilter , GrayscaleDilateImageFilter , GrayscaleFunctionDilateImageFilter , BinaryDilateImageFilter , ReconstructionByErosionImageFilter , OpeningByReconstructionImageFilter , ClosingByReconstructionImageFilter , ReconstructionImageFilter

     itk::simple::ReconstructionByErosion for the procedural interface

     itk::ReconstructionByErosionImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkReconstructionByErosionImageFilter.h

    
### sitk.RecursiveGaussian
    RecursiveGaussian(Image image1, double sigma=1.0, bool normalizeAcrossScale=False, itk::simple::RecursiveGaussianImageFilter::OrderType order, unsigned int direction=0) -> Image



    Base class for computing IIR convolution with an approximation of a
    Gaussian kernel.


    This function directly calls the execute method of RecursiveGaussianImageFilter in order to support a procedural API


    See:
     itk::simple::RecursiveGaussianImageFilter for the object oriented interface



    
### sitk.RecursiveGaussianImageFilter


    Base class for computing IIR convolution with an approximation of a
    Gaussian kernel.


    \[ \frac{ 1 }{ \sigma \sqrt{ 2 \pi } } \exp{
    \left( - \frac{x^2}{ 2 \sigma^2 } \right) } \]

    RecursiveGaussianImageFilter is the base class for recursive filters that approximate convolution
    with the Gaussian kernel. This class implements the recursive
    filtering method proposed by R.Deriche in IEEE-PAMI Vol.12, No.1,
    January 1990, pp 78-87, "Fast Algorithms for Low-Level Vision"

    Details of the implementation are described in the technical report: R.
    Deriche, "Recursively Implementing The Gaussian and Its
    Derivatives", INRIA, 1993, ftp://ftp.inria.fr/INRIA/tech-reports/RR/RR-1893.ps.gz

    Further improvements of the algorithm are described in: G. Farneback &
    C.-F. Westin, "On Implementation of Recursive Gaussian Filters", so
    far unpublished.

    As compared to itk::DiscreteGaussianImageFilter , this filter tends to be faster for large kernels, and it can take
    the derivative of the blurred image in one step. Also, note that we
    have itk::RecursiveGaussianImageFilter::SetSigma() , but itk::DiscreteGaussianImageFilter::SetVariance() .


    See:
     DiscreteGaussianImageFilter

     itk::simple::RecursiveGaussian for the procedural interface

     itk::RecursiveGaussianImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkRecursiveGaussianImageFilter.h

    
### sitk.RegionOfInterest
    RegionOfInterest(Image image1, VectorUInt32 size, VectorInt32 index) -> Image



    Extract a region of interest from the input image.


    This function directly calls the execute method of RegionOfInterestImageFilter in order to support a procedural API


    See:
     itk::simple::RegionOfInterestImageFilter for the object oriented interface



    
### sitk.RegionOfInterestImageFilter


    Extract a region of interest from the input image.


    This filter produces an output image of the same dimension as the
    input image. The user specifies the region of the input image that
    will be contained in the output image. The origin coordinates of the
    output images will be computed in such a way that if mapped to
    physical space, the output image will overlay the input image with
    perfect registration. In other words, a registration process between
    the output image and the input image will return an identity
    transform.

    If you are interested in changing the dimension of the image, you may
    want to consider the ExtractImageFilter . For example for extracting a 2D image from a slice of a 3D image.

    The region to extract is set using the method SetRegionOfInterest.


    See:
     ExtractImageFilter

     itk::simple::RegionOfInterest for the procedural interface

     itk::RegionOfInterestImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkRegionOfInterestImageFilter.h

    
### sitk.RegionalMaxima
    RegionalMaxima(Image image1, double backgroundValue=0.0, double foregroundValue=1.0, bool fullyConnected=False, bool flatIsMaxima=True) -> Image



    Produce a binary image where foreground is the regional maxima of the
    input image.


    This function directly calls the execute method of RegionalMaximaImageFilter in order to support a procedural API


    See:
     itk::simple::RegionalMaximaImageFilter for the object oriented interface



    
### sitk.RegionalMaximaImageFilter


    Produce a binary image where foreground is the regional maxima of the
    input image.


    Regional maxima are flat zones surrounded by pixels of lower value.

    If the input image is constant, the entire image can be considered as
    a maxima or not. The desired behavior can be selected with the SetFlatIsMaxima() method.


    Gaetan Lehmann
     This class was contributed to the Insight Journal by author Gaetan
    Lehmann. Biologie du Developpement et de la Reproduction, INRA de
    Jouy-en-Josas, France. The paper can be found at https://hdl.handle.net/1926/153


    See:
     ValuedRegionalMaximaImageFilter

     HConvexImageFilter

     RegionalMinimaImageFilter

     itk::simple::RegionalMaxima for the procedural interface

     itk::RegionalMaximaImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkRegionalMaximaImageFilter.h

    
### sitk.RegionalMinima
    RegionalMinima(Image image1, double backgroundValue=0.0, double foregroundValue=1.0, bool fullyConnected=False, bool flatIsMinima=True) -> Image



    Produce a binary image where foreground is the regional minima of the
    input image.


    This function directly calls the execute method of RegionalMinimaImageFilter in order to support a procedural API


    See:
     itk::simple::RegionalMinimaImageFilter for the object oriented interface



    
### sitk.RegionalMinimaImageFilter


    Produce a binary image where foreground is the regional minima of the
    input image.


    Regional minima are flat zones surrounded by pixels of greater value.

    If the input image is constant, the entire image can be considered as
    a minima or not. The SetFlatIsMinima() method let the user choose which behavior to use.

    This class was contribtued to the Insight Journal by
    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France. https://hdl.handle.net/1926/153

    See:
     RegionalMaximaImageFilter

     ValuedRegionalMinimaImageFilter

     HConcaveImageFilter

     itk::simple::RegionalMinima for the procedural interface

     itk::RegionalMinimaImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkRegionalMinimaImageFilter.h

    
### sitk.RelabelComponent
    RelabelComponent(Image image1, uint64_t minimumObjectSize=0, bool sortByObjectSize=True) -> Image



    Relabel the components in an image such that consecutive labels are
    used.


    This function directly calls the execute method of RelabelComponentImageFilter in order to support a procedural API


    See:
     itk::simple::RelabelComponentImageFilter for the object oriented interface



    
### sitk.RelabelComponentImageFilter


    Relabel the components in an image such that consecutive labels are
    used.


    RelabelComponentImageFilter remaps the labels associated with the objects in an image (as from
    the output of ConnectedComponentImageFilter ) such that the label numbers are consecutive with no gaps between
    the label numbers used. By default, the relabeling will also sort the
    labels based on the size of the object: the largest object will have
    label #1, the second largest will have label #2, etc. If two labels
    have the same size their initial order is kept. The sorting by size
    can be disabled using SetSortByObjectSize.

    Label #0 is assumed to be the background and is left unaltered by the
    relabeling.

    RelabelComponentImageFilter is typically used on the output of the ConnectedComponentImageFilter for those applications that want to extract the largest object or the
    "k" largest objects. Any particular object can be extracted from the
    relabeled output using a BinaryThresholdImageFilter . A group of objects can be extracted from the relabled output using
    a ThresholdImageFilter .

    Once all the objects are relabeled, the application can query the
    number of objects and the size of each object. Object sizes are returned in a vector. The size of the background is not
    calculated. So the size of object #1 is GetSizeOfObjectsInPixels() [0], the size of object #2 is GetSizeOfObjectsInPixels() [1], etc.

    If user sets a minimum object size, all objects with fewer pixels than
    the minimum will be discarded, so that the number of objects reported
    will be only those remaining. The GetOriginalNumberOfObjects method
    can be called to find out how many objects were present before the
    small ones were discarded.

    RelabelComponentImageFilter can be run as an "in place" filter, where it will overwrite its
    output. The default is run out of place (or generate a separate
    output). "In place" operation can be controlled via methods in the
    superclass, InPlaceImageFilter::InPlaceOn() and
    InPlaceImageFilter::InPlaceOff() .


    See:
     ConnectedComponentImageFilter , BinaryThresholdImageFilter , ThresholdImageFilter

     itk::simple::RelabelComponent for the procedural interface

     itk::RelabelComponentImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkRelabelComponentImageFilter.h

    
### sitk.RelabelLabelMap
    RelabelLabelMap(Image image1, bool reverseOrdering=True) -> Image



    This filter relabels the LabelObjects; the new labels are arranged
    consecutively with consideration for the background value.


    This function directly calls the execute method of RelabelLabelMapFilter in order to support a procedural API


    See:
     itk::simple::RelabelLabelMapFilter for the object oriented interface



    
### sitk.RelabelLabelMapFilter


    This filter relabels the LabelObjects; the new labels are arranged
    consecutively with consideration for the background value.


    This filter takes the LabelObjects from the input and reassigns them
    to the output by calling the PushLabelObject method, which by default,
    attempts to reorganize the labels consecutively. The user can assign
    an arbitrary value to the background; the filter will assign the
    labels consecutively by skipping the background value.

    This implementation was taken from the Insight Journal paper: https://hdl.handle.net/1926/584 or http://www.insight-journal.org/browse/publication/176
    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.

    See:
     ShapeLabelObject , RelabelComponentImageFilter

     itk::simple::RelabelLabelMapFilter for the procedural interface

     itk::RelabelLabelMapFilter for the Doxygen on the original ITK class.


    C++ includes: sitkRelabelLabelMapFilter.h

    
### sitk.RenyiEntropyThreshold
    RenyiEntropyThreshold(Image image, Image maskImage, uint8_t insideValue=1, uint8_t outsideValue=0, uint32_t numberOfHistogramBins=256, bool maskOutput=True, uint8_t maskValue=255) -> Image
    RenyiEntropyThreshold(Image image, uint8_t insideValue=1, uint8_t outsideValue=0, uint32_t numberOfHistogramBins=256, bool maskOutput=True, uint8_t maskValue=255) -> Image



    
### sitk.RenyiEntropyThresholdImageFilter


    Threshold an image using the RenyiEntropy Threshold.


    This filter creates a binary thresholded image that separates an image
    into foreground and background components. The filter computes the
    threshold using the RenyiEntropyThresholdCalculator and applies that threshold to the input image using the BinaryThresholdImageFilter .


    Richard Beare. Department of Medicine, Monash University, Melbourne,
    Australia.

    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.
     This implementation was taken from the Insight Journal paper: https://hdl.handle.net/10380/3279 or http://www.insight-journal.org/browse/publication/811


    See:
     HistogramThresholdImageFilter

     itk::simple::RenyiEntropyThreshold for the procedural interface

     itk::RenyiEntropyThresholdImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkRenyiEntropyThresholdImageFilter.h

    
### sitk.Resample
    Resample(Image image1, Transform transform, itk::simple::InterpolatorEnum interpolator, double defaultPixelValue=0.0, itk::simple::PixelIDValueEnum outputPixelType) -> Image
    Resample(Image image1, Image referenceImage, Transform transform, itk::simple::InterpolatorEnum interpolator, double defaultPixelValue=0.0, itk::simple::PixelIDValueEnum outputPixelType) -> Image
    Resample(Image image1, VectorUInt32 size, Transform transform, itk::simple::InterpolatorEnum interpolator, VectorDouble outputOrigin, VectorDouble outputSpacing, VectorDouble outputDirection, double defaultPixelValue=0.0, itk::simple::PixelIDValueEnum outputPixelType) -> Image
    
### sitk.ResampleImageFilter


    Resample an image via a coordinate transform.


    ResampleImageFilter resamples an existing image through some coordinate transform,
    interpolating via some image function. The class is templated over the
    types of the input and output images.

    Note that the choice of interpolator function can be important. This
    function is set via SetInterpolator() . The default is LinearInterpolateImageFunction <InputImageType, TInterpolatorPrecisionType>, which is reasonable for
    ordinary medical images. However, some synthetic images have pixels
    drawn from a finite prescribed set. An example would be a mask
    indicating the segmentation of a brain into a small number of tissue
    types. For such an image, one does not want to interpolate between
    different pixel values, and so NearestNeighborInterpolateImageFunction < InputImageType, TCoordRep > would be a better choice.

    If an sample is taken from outside the image domain, the default
    behavior is to use a default pixel value. If different behavior is
    desired, an extrapolator function can be set with SetExtrapolator() .

    Output information (spacing, size and direction) for the output image
    should be set. This information has the normal defaults of unit
    spacing, zero origin and identity direction. Optionally, the output
    information can be obtained from a reference image. If the reference
    image is provided and UseReferenceImage is On, then the spacing,
    origin and direction of the reference image will be used.

    Since this filter produces an image which is a different size than its
    input, it needs to override several of the methods defined in ProcessObject in order to properly manage the pipeline execution model. In
    particular, this filter overrides
    ProcessObject::GenerateInputRequestedRegion() and
    ProcessObject::GenerateOutputInformation() .

    This filter is implemented as a multithreaded filter. It provides a
    ThreadedGenerateData() method for its implementation.
    WARNING:
    For multithreading, the TransformPoint method of the user-designated
    coordinate transform must be threadsafe.

    See:
     itk::ResampleImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkResampleImageFilter.h

    
### sitk.RescaleIntensity
    RescaleIntensity(Image image1, double outputMinimum=0, double outputMaximum=255) -> Image



    Applies a linear transformation to the intensity levels of the input Image .


    This function directly calls the execute method of RescaleIntensityImageFilter in order to support a procedural API


    See:
     itk::simple::RescaleIntensityImageFilter for the object oriented interface



    
### sitk.RescaleIntensityImageFilter


    Applies a linear transformation to the intensity levels of the input Image .


    RescaleIntensityImageFilter applies pixel-wise a linear transformation to the intensity values of
    input image pixels. The linear transformation is defined by the user
    in terms of the minimum and maximum values that the output image
    should have.

    The following equation gives the mapping of the intensity values


    \[ outputPixel = ( inputPixel - inputMin) \cdot
    \frac{(outputMax - outputMin )}{(inputMax - inputMin)} + outputMin
    \]
     All computations are performed in the precision of the input pixel's
    RealType. Before assigning the computed value to the output pixel.

    NOTE: In this filter the minimum and maximum values of the input image
    are computed internally using the MinimumMaximumImageCalculator . Users are not supposed to set those values in this filter. If you
    need a filter where you can set the minimum and maximum values of the
    input, please use the IntensityWindowingImageFilter . If you want a filter that can use a user-defined linear
    transformation for the intensity, then please use the ShiftScaleImageFilter .


    See:
     IntensityWindowingImageFilter

     itk::simple::RescaleIntensity for the procedural interface

     itk::RescaleIntensityImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkRescaleIntensityImageFilter.h

    
### sitk.RichardsonLucyDeconvolution
    RichardsonLucyDeconvolution(Image image1, Image image2, int numberOfIterations=1, bool normalize=False, itk::simple::RichardsonLucyDeconvolutionImageFilter::BoundaryConditionType boundaryCondition, itk::simple::RichardsonLucyDeconvolutionImageFilter::OutputRegionModeType outputRegionMode) -> Image



    Deconvolve an image using the Richardson-Lucy deconvolution algorithm.


    This function directly calls the execute method of RichardsonLucyDeconvolutionImageFilter in order to support a procedural API


    See:
     itk::simple::RichardsonLucyDeconvolutionImageFilter for the object oriented interface



    
### sitk.RichardsonLucyDeconvolutionImageFilter


    Deconvolve an image using the Richardson-Lucy deconvolution algorithm.


    This filter implements the Richardson-Lucy deconvolution algorithm as
    defined in Bertero M and Boccacci P, "Introduction to Inverse
    Problems in Imaging", 1998. The algorithm assumes that the input
    image has been formed by a linear shift-invariant system with a known
    kernel.

    The Richardson-Lucy algorithm assumes that noise in the image follows
    a Poisson distribution and that the distribution for each pixel is
    independent of the other pixels.

    This code was adapted from the Insight Journal contribution:

    "Deconvolution: infrastructure and reference algorithms" by Gaetan
    Lehmann https://hdl.handle.net/10380/3207


    Gaetan Lehmann, Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France

    Cory Quammen, The University of North Carolina at Chapel Hill

    See:
     IterativeDeconvolutionImageFilter

     LandweberDeconvolutionImageFilter

     ProjectedLandweberDeconvolutionImageFilter

     itk::simple::RichardsonLucyDeconvolution for the procedural interface

     itk::RichardsonLucyDeconvolutionImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkRichardsonLucyDeconvolutionImageFilter.h

    
### sitk.Round
    Round(Image image1) -> Image



    Rounds the value of each pixel.


    This function directly calls the execute method of RoundImageFilter in order to support a procedural API


    See:
     itk::simple::RoundImageFilter for the object oriented interface



    
### sitk.RoundImageFilter


    Rounds the value of each pixel.


    The computations are performed using itk::Math::Round(x).
    See:
     itk::simple::Round for the procedural interface

     itk::RoundImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkRoundImageFilter.h

    
### sitk.SITK_ITK_VERSION_MAJORint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.SITK_ITK_VERSION_MINORint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.SITK_ITK_VERSION_PATCHint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.SLIC
    SLIC(Image image1, VectorUInt32 superGridSize, double spatialProximityWeight=10.0, uint32_t maximumNumberOfIterations=5, bool enforceConnectivity=True, bool initializationPerturbation=True) -> Image



    Simple Linear Iterative Clustering (SLIC) super-pixel segmentation.


    This function directly calls the execute method of SLICImageFilter in order to support a procedural API


    See:
     itk::simple::SLICImageFilter for the object oriented interface



    
### sitk.SLICImageFilter


    Simple Linear Iterative Clustering (SLIC) super-pixel segmentation.


    The Simple Linear Iterative Clustering (SLIC) algorithm groups pixels
    into a set of labeled regions or super-pixels. Super-pixels follow
    natural image boundaries, are compact, and are nearly uniform regions
    which can be used as a larger primitive for more efficient
    computation. The SLIC algorithm can be viewed as a spatially
    constrained iterative k-means method.

    The original algorithm was designed to cluster on the joint domain of
    the images index space and it's CIELAB color space. This
    implementation works with images of arbitrary dimension as well as
    scalar, single channel, images and most multi-component image types
    including ITK's arbitrary length VectorImage .

    The distance between a pixel and a cluster is the sum of squares of
    the difference between their joint range and domains ( index and value
    ). The computation is done in index space with scales provided by the
    SpatialProximityWeight parameters.

    The output is a label image with each label representing a superpixel
    cluster. Every pixel in the output is labeled, and the starting label
    id is zero.

    This code was contributed in the Insight Journal paper: "Scalable
    Simple Linear Iterative Clustering (SSLIC) Using a Generic and
    Parallel Approach" by Lowekamp B. C., Chen D. T., Yaniv Z.
    See:
     itk::simple::SLIC for the procedural interface

     itk::SLICImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkSLICImageFilter.h

    
### sitk.STAPLE
    STAPLE(VectorOfImage images, double confidenceWeight=1.0, double foregroundValue=1.0, unsigned int maximumIterations) -> Image
    STAPLE(Image image1, double confidenceWeight=1.0, double foregroundValue=1.0, unsigned int maximumIterations) -> Image
    STAPLE(Image image1, Image image2, double confidenceWeight=1.0, double foregroundValue=1.0, unsigned int maximumIterations) -> Image
    STAPLE(Image image1, Image image2, Image image3, double confidenceWeight=1.0, double foregroundValue=1.0, unsigned int maximumIterations) -> Image
    STAPLE(Image image1, Image image2, Image image3, Image image4, double confidenceWeight=1.0, double foregroundValue=1.0, unsigned int maximumIterations) -> Image
    STAPLE(Image image1, Image image2, Image image3, Image image4, Image image5, double confidenceWeight=1.0, double foregroundValue=1.0, unsigned int maximumIterations) -> Image
    
### sitk.STAPLEImageFilter


    The STAPLE filter implements the Simultaneous Truth and Performance
    Level Estimation algorithm for generating ground truth volumes from a
    set of binary expert segmentations.


    The STAPLE algorithm treats segmentation as a pixelwise
    classification, which leads to an averaging scheme that accounts for
    systematic biases in the behavior of experts in order to generate a
    fuzzy ground truth volume and simultaneous accuracy assessment of each
    expert. The ground truth volumes produced by this filter are floating
    point volumes of values between zero and one that indicate probability
    of each pixel being in the object targeted by the segmentation.

    The STAPLE algorithm is described in

    S. Warfield, K. Zou, W. Wells, "Validation of image segmentation and
    expert quality with an expectation-maximization algorithm" in MICCAI
    2002: Fifth International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer-Verlag,
    Heidelberg, Germany, 2002, pp. 298-306

    INPUTS
    Input volumes to the STAPLE filter must be binary segmentations of an
    image, that is, there must be a single foreground value that
    represents positively classified pixels (pixels that are considered to
    belong inside the segmentation). Any number of background pixel values
    may be present in the input images. You can, for example, input
    volumes with many different labels as long as the structure you are
    interested in creating ground truth for is consistently labeled among
    all input volumes. Pixel type of the input volumes does not matter.
    Specify the label value for positively classified pixels using
    SetForegroundValue. All other labels will be considered to be
    negatively classified pixels (background).
     Input volumes must all contain the same size RequestedRegions.

    OUTPUTS
    The STAPLE filter produces a single output volume with a range of
    floating point values from zero to one. IT IS VERY IMPORTANT TO
    INSTANTIATE THIS FILTER WITH A FLOATING POINT OUTPUT TYPE (floats or
    doubles). You may threshold the output above some probability
    threshold if you wish to produce a binary ground truth.
    PARAMETERS
    The STAPLE algorithm requires a number of inputs. You may specify any
    number of input volumes using the SetInput(i, p_i) method, where i
    ranges from zero to N-1, N is the total number of input segmentations,
    and p_i is the SmartPointer to the i-th segmentation.
     The SetConfidenceWeight parameter is a modifier for the prior
    probability that any pixel would be classified as inside the target
    object. This implementation of the STAPLE algorithm automatically
    calculates prior positive classification probability as the average
    fraction of the image volume filled by the target object in each input
    segmentation. The ConfidenceWeight parameter allows for scaling the of
    this default prior probability: if g_t is the prior probability that a
    pixel would be classified inside the target object, then g_t is set to
    g_t * ConfidenceWeight before iterating on the solution. In general
    ConfidenceWeight should be left to the default of 1.0.

    You must provide a foreground value using SetForegroundValue that the
    STAPLE algorithm will use to identify positively classified pixels in
    the the input images. All other values in the image will be treated as
    background values. For example, if your input segmentations consist of
    1's everywhere inside the segmented region, then use
    SetForegroundValue(1).

    The STAPLE algorithm is an iterative E-M algorithm and will converge
    on a solution after some number of iterations that cannot be known a
    priori. After updating the filter, the total elapsed iterations taken
    to converge on the solution can be queried through GetElapsedIterations() . You may also specify a MaximumNumberOfIterations, after which the
    algorithm will stop iterating regardless of whether or not it has
    converged. This implementation of the STAPLE algorithm will find the
    solution to within seven digits of precision unless it is stopped
    early.

    Once updated, the Sensitivity (true positive fraction, q) and
    Specificity (true negative fraction, q) for each expert input volume
    can be queried using GetSensitivity(i) and GetSpecificity(i), where i
    is the i-th input volume.

    REQUIRED PARAMETERS
    The only required parameters for this filter are the ForegroundValue
    and the input volumes. All other parameters may be safely left to
    their default values. Please see the paper cited above for more
    information on the STAPLE algorithm and its parameters. A proper
    understanding of the algorithm is important for interpreting the
    results that it produces.
    EVENTS
    This filter invokes IterationEvent() at each iteration of the E-M
    algorithm. Setting the AbortGenerateData() flag will cause the
    algorithm to halt after the current iteration and produce results just
    as if it had converged. The algorithm makes no attempt to report its
    progress since the number of iterations needed cannot be known in
    advance.

    See:
     itk::simple::STAPLE for the procedural interface


    C++ includes: sitkSTAPLEImageFilter.h

    
### sitk.SaltAndPepperNoise
    SaltAndPepperNoise(Image image1, double probability=0.01, uint32_t seed) -> Image



    Alter an image with fixed value impulse noise, often called salt and
    pepper noise.


    This function directly calls the execute method of SaltAndPepperNoiseImageFilter in order to support a procedural API


    See:
     itk::simple::SaltAndPepperNoiseImageFilter for the object oriented interface



    
### sitk.SaltAndPepperNoiseImageFilter


    Alter an image with fixed value impulse noise, often called salt and
    pepper noise.


    Salt and pepper noise is a special kind of impulse noise where the
    value of the noise is either the maximum possible value in the image
    or its minimum. It can be modeled as:


    $ I = \begin{cases} M, & \quad \text{if } U < p/2 \\ m,
    & \quad \text{if } U > 1 - p/2 \\ I_0, & \quad
    \text{if } p/2 \geq U \leq 1 - p/2 \end{cases} $

    where $ p $ is the probability of the noise event, $ U $ is a uniformly distributed random variable in the $ [0,1] $ range, $ M $ is the greatest possible pixel value, and $ m $ the smallest possible pixel value.
     Pixel alteration occurs at a user defined probability. Salt and
    pepper pixels are equally distributed.


    Gaetan Lehmann
     This code was contributed in the Insight Journal paper "Noise
    Simulation". https://hdl.handle.net/10380/3158
    See:
     itk::simple::SaltAndPepperNoise for the procedural interface

     itk::SaltAndPepperNoiseImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkSaltAndPepperNoiseImageFilter.h

    
### sitk.ScalarChanAndVeseDenseLevelSet
    ScalarChanAndVeseDenseLevelSet(Image image1, Image image2, double maximumRMSError=0.02, uint32_t numberOfIterations=1000, double lambda1=1.0, double lambda2=1.0, double epsilon=1.0, double curvatureWeight=1.0, double areaWeight=0.0, double reinitializationSmoothingWeight=0.0, double volume=0.0, double volumeMatchingWeight=0.0, itk::simple::ScalarChanAndVeseDenseLevelSetImageFilter::HeavisideStepFunctionType heavisideStepFunction, bool useImageSpacing=True) -> Image



    Dense implementation of the Chan and Vese multiphase level set image
    filter.


    This function directly calls the execute method of ScalarChanAndVeseDenseLevelSetImageFilter in order to support a procedural API


    See:
     itk::simple::ScalarChanAndVeseDenseLevelSetImageFilter for the object oriented interface



    
### sitk.ScalarChanAndVeseDenseLevelSetImageFilter


    Dense implementation of the Chan and Vese multiphase level set image
    filter.


    This code was adapted from the paper: "An active contour model
    without edges" T. Chan and L. Vese. In Scale-Space Theories in
    Computer Vision, pages 141-151, 1999.


    Mosaliganti K., Smith B., Gelas A., Gouaillard A., Megason S.
     This code was taken from the Insight Journal paper: "Cell Tracking
    using Coupled Active Surfaces for Nuclei and Membranes" http://www.insight-journal.org/browse/publication/642 https://hdl.handle.net/10380/3055

    That is based on the papers: "Level Set Segmentation: Active Contours
    without edge" http://www.insight-journal.org/browse/publication/322 https://hdl.handle.net/1926/1532

    and

    "Level set segmentation using coupled active surfaces" http://www.insight-journal.org/browse/publication/323 https://hdl.handle.net/1926/1533
    See:
     itk::simple::ScalarChanAndVeseDenseLevelSet for the procedural interface

     itk::ScalarChanAndVeseDenseLevelSetImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkScalarChanAndVeseDenseLevelSetImageFilter.h

    
### sitk.ScalarConnectedComponent
    ScalarConnectedComponent(Image image, Image maskImage, double distanceThreshold=0.0, bool fullyConnected=False) -> Image
    ScalarConnectedComponent(Image image, double distanceThreshold=0.0, bool fullyConnected=False) -> Image



    
### sitk.ScalarConnectedComponentImageFilter


    A connected components filter that labels the objects in an arbitrary
    image. Two pixels are similar if they are within threshold of each
    other. Uses ConnectedComponentFunctorImageFilter .



    See:
     itk::simple::ScalarConnectedComponent for the procedural interface

     itk::ScalarConnectedComponentImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkScalarConnectedComponentImageFilter.h

    
### sitk.ScalarImageKmeans
    ScalarImageKmeans(Image image1, VectorDouble classWithInitialMean, bool useNonContiguousLabels=False) -> Image



    Classifies the intensity values of a scalar image using the K-Means
    algorithm.


    This function directly calls the execute method of ScalarImageKmeansImageFilter in order to support a procedural API


    See:
     itk::simple::ScalarImageKmeansImageFilter for the object oriented interface



    
### sitk.ScalarImageKmeansImageFilter


    Classifies the intensity values of a scalar image using the K-Means
    algorithm.


    Given an input image with scalar values, it uses the K-Means
    statistical classifier in order to define labels for every pixel in
    the image. The filter is templated over the type of the input image.
    The output image is predefined as having the same dimension of the
    input image and pixel type unsigned char, under the assumption that
    the classifier will generate less than 256 classes.

    You may want to look also at the RelabelImageFilter that may be used
    as a postprocessing stage, in particular if you are interested in
    ordering the labels by their relative size in number of pixels.


    See:
     Image

     ImageKmeansModelEstimator

     KdTreeBasedKmeansEstimator, WeightedCentroidKdTreeGenerator, KdTree

     RelabelImageFilter

     itk::simple::ScalarImageKmeans for the procedural interface

     itk::ScalarImageKmeansImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkScalarImageKmeansImageFilter.h

    
### sitk.ScalarToRGBColormap
    ScalarToRGBColormap(Image image1, itk::simple::ScalarToRGBColormapImageFilter::ColormapType colormap, bool useInputImageExtremaForScaling=True) -> Image



    Implements pixel-wise intensity->rgb mapping operation on one image.


    This function directly calls the execute method of ScalarToRGBColormapImageFilter in order to support a procedural API


    See:
     itk::simple::ScalarToRGBColormapImageFilter for the object oriented interface



    
### sitk.ScalarToRGBColormapImageFilter


    Implements pixel-wise intensity->rgb mapping operation on one image.


    This class is parameterized over the type of the input image and the
    type of the output image.

    The input image's scalar pixel values are mapped into a color map. The
    color map is specified by passing the SetColormap function one of the
    predefined maps. The following selects the "Hot" colormap:

    You can also specify a custom color map. This is done by creating a
    CustomColormapFunction, and then creating lists of values for the red,
    green, and blue channel. An example of setting the red channel of a
    colormap with only 2 colors is given below. The blue and green
    channels should be specified in the same manner.


    The range of values present in the input image is the range that is
    mapped to the entire range of colors.

    This code was contributed in the Insight Journal paper: "Meeting Andy
    Warhol Somewhere Over the Rainbow: RGB Colormapping and ITK" by
    Tustison N., Zhang H., Lehmann G., Yushkevich P., Gee J. https://hdl.handle.net/1926/1452 http://www.insight-journal.org/browse/publication/285


    See:
     BinaryFunctionImageFilter TernaryFunctionImageFilter

     itk::simple::ScalarToRGBColormap for the procedural interface

     itk::ScalarToRGBColormapImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkScalarToRGBColormapImageFilter.h

    
### sitk.ScaleSkewVersor3DTransform


    A over parameterized 3D Affine transform composed of the addition of a
    versor rotation matrix, a scale matrix and a skew matrix around a
    fixed center with translation.



    See:
     itk::ScaleSkewVersor3DTransform


    C++ includes: sitkScaleSkewVersor3DTransform.h

    
### sitk.ScaleTransform


    A 2D or 3D anisotropic scale of coordinate space around a fixed
    center.



    See:
     itk::ScaleTransform


    C++ includes: sitkScaleTransform.h

    
### sitk.ScaleVersor3DTransform


    A parameterized 3D transform composed of the addition of a versor
    rotation matrix and a scale matrix around a fixed center with
    translation.



    See:
     itk::ScaleVersor3DTransform


    C++ includes: sitkScaleVersor3DTransform.h

    
### sitk.ShanbhagThreshold
    ShanbhagThreshold(Image image, Image maskImage, uint8_t insideValue=1, uint8_t outsideValue=0, uint32_t numberOfHistogramBins=256, bool maskOutput=True, uint8_t maskValue=255) -> Image
    ShanbhagThreshold(Image image, uint8_t insideValue=1, uint8_t outsideValue=0, uint32_t numberOfHistogramBins=256, bool maskOutput=True, uint8_t maskValue=255) -> Image



    
### sitk.ShanbhagThresholdImageFilter


    Threshold an image using the Shanbhag Threshold.


    This filter creates a binary thresholded image that separates an image
    into foreground and background components. The filter computes the
    threshold using the ShanbhagThresholdCalculator and applies that threshold to the input image using the BinaryThresholdImageFilter .


    Richard Beare. Department of Medicine, Monash University, Melbourne,
    Australia.

    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.
     This implementation was taken from the Insight Journal paper: https://hdl.handle.net/10380/3279 or http://www.insight-journal.org/browse/publication/811


    See:
     HistogramThresholdImageFilter

     itk::simple::ShanbhagThreshold for the procedural interface

     itk::ShanbhagThresholdImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkShanbhagThresholdImageFilter.h

    
### sitk.ShapeDetectionLevelSet
    ShapeDetectionLevelSet(Image image1, Image image2, double maximumRMSError=0.02, double propagationScaling=1.0, double curvatureScaling=1.0, uint32_t numberOfIterations=1000, bool reverseExpansionDirection=False) -> Image



    Segments structures in images based on a user supplied edge potential
    map.


    This function directly calls the execute method of ShapeDetectionLevelSetImageFilter in order to support a procedural API


    See:
     itk::simple::ShapeDetectionLevelSetImageFilter for the object oriented interface



    
### sitk.ShapeDetectionLevelSetImageFilter


    Segments structures in images based on a user supplied edge potential
    map.


    IMPORTANT
    The SegmentationLevelSetImageFilter class and the ShapeDetectionLevelSetFunction class contain additional information necessary to gain full
    understanding of how to use this filter.
    OVERVIEW
    This class is a level set method segmentation filter. An initial
    contour is propagated outwards (or inwards) until it ''sticks'' to the
    shape boundaries. This is done by using a level set speed function
    based on a user supplied edge potential map. This approach for
    segmentation follows that of Malladi et al (1995).
    INPUTS
    This filter requires two inputs. The first input is a initial level
    set. The initial level set is a real image which contains the initial
    contour/surface as the zero level set. For example, a signed distance
    function from the initial contour/surface is typically used. Note that
    for this algorithm the initial contour has to be wholly within (or
    wholly outside) the structure to be segmented.

    The second input is the feature image. For this filter, this is the
    edge potential map. General characteristics of an edge potential map
    is that it has values close to zero in regions near the edges and
    values close to one inside the shape itself. Typically, the edge
    potential map is compute from the image gradient, for example:
    \[ g(I) = 1 / ( 1 + | (\nabla * G)(I)| ) \] \[ g(I) = \exp^{-|(\nabla * G)(I)|} \]

    where $ I $ is image intensity and $ (\nabla * G) $ is the derivative of Gaussian operator.


    See SegmentationLevelSetImageFilter and SparseFieldLevelSetImageFilter for more information on Inputs.
    PARAMETERS
    The PropagationScaling parameter can be used to switch from
    propagation outwards (POSITIVE scaling parameter) versus propagating
    inwards (NEGATIVE scaling parameter).
     The smoothness of the resulting contour/surface can be adjusted using
    a combination of PropagationScaling and CurvatureScaling parameters.
    The larger the CurvatureScaling parameter, the smoother the resulting
    contour. The CurvatureScaling parameter should be non-negative for
    proper operation of this algorithm. To follow the implementation in
    Malladi et al paper, set the PropagtionScaling to $\pm 1.0$ and CurvatureScaling to $ \epsilon $ .

    Note that there is no advection term for this filter. Setting the
    advection scaling will have no effect.

    OUTPUTS
    The filter outputs a single, scalar, real-valued image. Negative
    values in the output image represent the inside of the segmentated
    region and positive values in the image represent the outside of the
    segmented region. The zero crossings of the image correspond to the
    position of the propagating front.

    See SparseFieldLevelSetImageFilter and SegmentationLevelSetImageFilter for more information.
    REFERENCES

    "Shape Modeling with Front Propagation: A Level Set Approach", R.
    Malladi, J. A. Sethian and B. C. Vermuri. IEEE Trans. on Pattern
    Analysis and Machine Intelligence, Vol 17, No. 2, pp 158-174, February
    1995

    See:
     SegmentationLevelSetImageFilter

     ShapeDetectionLevelSetFunction

     SparseFieldLevelSetImageFilter

     itk::simple::ShapeDetectionLevelSet for the procedural interface

     itk::ShapeDetectionLevelSetImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkShapeDetectionLevelSetImageFilter.h

    
### sitk.ShiftScale
    ShiftScale(Image image1, double shift=0, double scale=1.0) -> Image



    Shift and scale the pixels in an image.


    This function directly calls the execute method of ShiftScaleImageFilter in order to support a procedural API


    See:
     itk::simple::ShiftScaleImageFilter for the object oriented interface



    
### sitk.ShiftScaleImageFilter


    Shift and scale the pixels in an image.


    ShiftScaleImageFilter shifts the input pixel by Shift (default 0.0) and then scales the
    pixel by Scale (default 1.0). All computattions are performed in the
    precision of the input pixel's RealType. Before assigning the computed
    value to the output pixel, the value is clamped at the NonpositiveMin
    and max of the pixel type.
    See:
     itk::simple::ShiftScale for the procedural interface

     itk::ShiftScaleImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkShiftScaleImageFilter.h

    
### sitk.ShotNoise
    ShotNoise(Image image1, double scale=1.0, uint32_t seed) -> Image



    Alter an image with shot noise.


    This function directly calls the execute method of ShotNoiseImageFilter in order to support a procedural API


    See:
     itk::simple::ShotNoiseImageFilter for the object oriented interface



    
### sitk.ShotNoiseImageFilter


    Alter an image with shot noise.


    The shot noise follows a Poisson distribution:


    $ I = N(I_0) $

    where $ N(I_0) $ is a Poisson-distributed random variable of mean $ I_0 $ . The noise is thus dependent on the pixel intensities in the image.
     The intensities in the image can be scaled by a user provided value
    to map pixel values to the actual number of particles. The scaling can
    be seen as the inverse of the gain used during the acquisition. The
    noisy signal is then scaled back to its input intensity range:


    $ I = \frac{N(I_0 \times s)}{s} $

    where $ s $ is the scale factor.
     The Poisson-distributed variable $ \lambda $ is computed by using the algorithm:


    $ \begin{array}{l} k \leftarrow 0 \\ p \leftarrow 1
    \\ \textbf{repeat} \\ \left\{ \begin{array}{l}
    k \leftarrow k+1 \\ p \leftarrow p \ast U()
    \end{array} \right. \\ \textbf{until } p >
    e^{\lambda} \\ \textbf{return} (k) \end{array} $

    where $ U() $ provides a uniformly distributed random variable in the interval $ [0,1] $ .
     This algorithm is very inefficient for large values of $ \lambda $ , though. Fortunately, the Poisson distribution can be accurately
    approximated by a Gaussian distribution of mean and variance $ \lambda $ when $ \lambda $ is large enough. In this implementation, this value is considered to
    be 50. This leads to the faster algorithm:


    $ \lambda + \sqrt{\lambda} \times N()$

    where $ N() $ is a normally distributed random variable of mean 0 and variance 1.

    Gaetan Lehmann
     This code was contributed in the Insight Journal paper "Noise
    Simulation". https://hdl.handle.net/10380/3158
    See:
     itk::simple::ShotNoise for the procedural interface

     itk::ShotNoiseImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkShotNoiseImageFilter.h

    
### sitk.Show
    Show(Image image, std::string const & title, bool const debugOn)



    Display an image using Fiji, ImageJ or another application.

    This function requires that Fiji ( https://fiji.sc ) or ImageJ ( http://rsb.info.nih.gov/ij/) be properly installed for Mac and Windows, and in the user's path
    for Linux. ImageJ must have a plugin for reading Nifti formatted files
    ( http://www.loci.wisc.edu/bio-formats/imagej).

    Nifti is the default file format used to export images. A different
    format can be chosen by setting the SITK_SHOW_EXTENSION environment
    variable. For example, set SITK_SHOW_EXTENSION to ".png" to use PNG
    format.

    The user can specify an application other than ImageJ to view images
    via the SITK_SHOW_COMMAND environment variable.

    The user can also select applications specifically for color images or
    3D images using the SITK_SHOW_COLOR_COMMAND and SITK_SHOW_3D_COMMAND
    environment variables.

    SITK_SHOW_COMMAND, SITK_SHOW_COLOR_COMMAND and SITK_SHOW_3D_COMMAND
    allow the following tokens in their strings.\li \c "%a"  for the ImageJ application \li \c "%f"
    for SimpleITK's temporary image file

    For example, the default SITK_SHOW_COMMAND string on Linux systems is:


    After token substitution it may become:


    For another example, the default SITK_SHOW_COLOR_COMMAND string on Mac
    OS X is:


    After token substitution the string may become:


    The string after "-eval" is an ImageJ macro the opens the file and runs ImageJ's Make
    Composite command to display the image in color.

    If the "%f" token is not found in the command string, the temporary file name is
    automatically appended to the command argument list.

    When invoked, Show searches for Fiji first, and then ImageJ. Fiji is
    the most update-to-date version of ImageJ and includes a lot of
    plugins which facilitate scientific image analysis. By default, for a
    64-bit build of SimpleITK on Macs, sitkShow searches for ImageJ64.app.
    For a 32-bit Mac build, sitkShow searches for ImageJ.app. If the user
    prefers a different version of ImageJ (or a different image viewer
    altogether), it can be specified using the SITK_SHOW_COMMAND
    environment variable.

    The boolean parameter debugOn prints the search path Show uses to find
    ImageJ, the full path to the ImageJ it found, and the full command
    line used to invoke ImageJ.


    
### sitk.Shrink
    Shrink(Image image1, VectorUInt32 shrinkFactors) -> Image



    Reduce the size of an image by an integer factor in each dimension.


    This function directly calls the execute method of ShrinkImageFilter in order to support a procedural API


    See:
     itk::simple::ShrinkImageFilter for the object oriented interface



    
### sitk.ShrinkImageFilter


    Reduce the size of an image by an integer factor in each dimension.


    ShrinkImageFilter reduces the size of an image by an integer factor in each dimension.
    The algorithm implemented is a simple subsample. The output image size
    in each dimension is given by:

    outputSize[j] = max( std::floor(inputSize[j]/shrinkFactor[j]), 1 );

    NOTE: The physical centers of the input and output will be the same.
    Because of this, the Origin of the output may not be the same as the
    Origin of the input. Since this filter produces an image which is a
    different resolution, origin and with different pixel spacing than its
    input image, it needs to override several of the methods defined in ProcessObject in order to properly manage the pipeline execution model. In
    particular, this filter overrides
    ProcessObject::GenerateInputRequestedRegion() and
    ProcessObject::GenerateOutputInformation() .

    This filter is implemented as a multithreaded filter. It provides a
    ThreadedGenerateData() method for its implementation.
    See:
     itk::simple::Shrink for the procedural interface

     itk::ShrinkImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkShrinkImageFilter.h

    
### sitk.Sigmoid
    Sigmoid(Image image1, double alpha=1, double beta=0, double outputMaximum=255, double outputMinimum=0) -> Image



    Computes the sigmoid function pixel-wise.


    This function directly calls the execute method of SigmoidImageFilter in order to support a procedural API


    See:
     itk::simple::SigmoidImageFilter for the object oriented interface



    
### sitk.SigmoidImageFilter


    Computes the sigmoid function pixel-wise.


    A linear transformation is applied first on the argument of the
    sigmoid function. The resulting total transform is given by

    \[ f(x) = (Max-Min) \cdot \frac{1}{\left(1+e^{- \frac{
    x - \beta }{\alpha}}\right)} + Min \]

    Every output pixel is equal to f(x). Where x is the intensity of the
    homologous input pixel, and alpha and beta are user-provided
    constants.
    See:
     itk::simple::Sigmoid for the procedural interface

     itk::SigmoidImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkSigmoidImageFilter.h

    
### sitk.SignedDanielssonDistanceMap
    SignedDanielssonDistanceMap(Image image1, bool insideIsPositive=False, bool squaredDistance=False, bool useImageSpacing=False) -> Image



    itk::simple::SignedDanielssonDistanceMapImageFilter Procedural Interface


    This function directly calls the execute method of SignedDanielssonDistanceMapImageFilter in order to support a procedural API


    See:
     itk::simple::SignedDanielssonDistanceMapImageFilter for the object oriented interface



    
### sitk.SignedDanielssonDistanceMapImageFilter


    This class is parametrized over the type of the input image and the
    type of the output image.

    This filter computes the distance map of the input image as an
    approximation with pixel accuracy to the Euclidean distance.

    For purposes of evaluating the signed distance map, the input is
    assumed to be binary composed of pixels with value 0 and non-zero.

    The inside is considered as having negative distances. Outside is
    treated as having positive distances. To change the convention, use
    the InsideIsPositive(bool) function.

    As a convention, the distance is evaluated from the boundary of the ON
    pixels.

    The filter returns


    A signed distance map with the approximation to the euclidean
    distance.

    A voronoi partition. (See itkDanielssonDistanceMapImageFilter)

    A vector map containing the component of the vector relating the
    current pixel with the closest point of the closest object to this
    pixel. Given that the components of the distance are computed in
    "pixels", the vector is represented by an itk::Offset . That is, physical coordinates are not used. (See
    itkDanielssonDistanceMapImageFilter)
     This filter internally uses the DanielssonDistanceMap filter. This
    filter is N-dimensional.


    See:
     itkDanielssonDistanceMapImageFilter

     itk::simple::SignedDanielssonDistanceMap for the procedural interface

     itk::SignedDanielssonDistanceMapImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkSignedDanielssonDistanceMapImageFilter.h

    
### sitk.SignedMaurerDistanceMap
    SignedMaurerDistanceMap(Image image1, bool insideIsPositive=False, bool squaredDistance=True, bool useImageSpacing=False, double backgroundValue=0.0) -> Image



    This filter calculates the Euclidean distance transform of a binary
    image in linear time for arbitrary dimensions.


    This function directly calls the execute method of SignedMaurerDistanceMapImageFilter in order to support a procedural API


    See:
     itk::simple::SignedMaurerDistanceMapImageFilter for the object oriented interface



    
### sitk.SignedMaurerDistanceMapImageFilter


    This filter calculates the Euclidean distance transform of a binary
    image in linear time for arbitrary dimensions.


    Inputs and Outputs
    This is an image-to-image filter. The dimensionality is arbitrary. The
    only dimensionality constraint is that the input and output images be
    of the same dimensions and size. To maintain integer arithmetic within
    the filter, the default output is the signed squared distance. This
    implies that the input image should be of type "unsigned int" or
    "int" whereas the output image is of type "int". Obviously, if the
    user wishes to utilize the image spacing or to have a filter with the
    Euclidean distance (as opposed to the squared distance), output image
    types of float or double should be used.
     The inside is considered as having negative distances. Outside is
    treated as having positive distances. To change the convention, use
    the InsideIsPositive(bool) function.

    Parameters
    Set/GetBackgroundValue specifies the background of the value of the
    input binary image. Normally this is zero and, as such, zero is the
    default value. Other than that, the usage is completely analogous to
    the itk::DanielssonDistanceImageFilter class except it does not return
    the Voronoi map.
     Reference: C. R. Maurer, Jr., R. Qi, and V. Raghavan, "A Linear Time
    Algorithm for Computing Exact Euclidean Distance Transforms of Binary
    Images in Arbitrary Dimensions", IEEE - Transactions on Pattern
    Analysis and Machine Intelligence, 25(2): 265-270, 2003.
    See:
     itk::simple::SignedMaurerDistanceMap for the procedural interface

     itk::SignedMaurerDistanceMapImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkSignedMaurerDistanceMapImageFilter.h

    
### sitk.Similarity2DTransform


    A similarity 2D transform with rotation in radians and isotropic
    scaling around a fixed center with translation.



    See:
     itk::Similarity2DTransform


    C++ includes: sitkSimilarity2DTransform.h

    
### sitk.Similarity3DTransform


    A similarity 3D transform with rotation as a versor, and isotropic
    scaling around a fixed center with translation.



    See:
     itk::Similarity3DTransform


    C++ includes: sitkSimilarity3DTransform.h

    
### sitk.SimilarityIndexImageFilter


    Measures the similarity between the set of non-zero pixels of two
    images.


    SimilarityIndexImageFilter measures the similarity between the set non-zero pixels of two images
    using the following formula: \[ S = \frac{2 | A \cap B |}{|A| + |B|} \] where $A$ and $B$ are respectively the set of non-zero pixels in the first and second
    input images. Operator $|\cdot|$ represents the size of a set and $\cap$ represents the intersection of two sets.

    The measure is derived from a reliability measure known as the kappa
    statistic. $S$ is sensitive to both differences in size and in location and have
    been in the literature for comparing two segmentation masks. For more
    information see: "Morphometric Analysis of White Matter Lesions in MR
    Images: Method and Validation", A. P. Zijdenbos, B. M. Dawant, R. A.
    Margolin and A. C. Palmer, IEEE Trans. on Medical Imaging, 13(4) pp
    716-724,1994

    This filter requires the largest possible region of the first image
    and the same corresponding region in the second image. It behaves as
    filter with two input and one output. Thus it can be inserted in a
    pipeline with other filters. The filter passes the first input through
    unmodified.

    This filter is templated over the two input image type. It assume both
    image have the same number of dimensions.


    See:
     itk::SimilarityIndexImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkSimilarityIndexImageFilter.h

    
### sitk.SimpleContourExtractor
    SimpleContourExtractor(Image image1, double inputForegroundValue=1.0, double inputBackgroundValue=0.0, VectorUInt32 radius, double outputForegroundValue=1.0, double outputBackgroundValue=0.0) -> Image



    Computes an image of contours which will be the contour of the first
    image.


    This function directly calls the execute method of SimpleContourExtractorImageFilter in order to support a procedural API


    See:
     itk::simple::SimpleContourExtractorImageFilter for the object oriented interface



    
### sitk.SimpleContourExtractorImageFilter


    Computes an image of contours which will be the contour of the first
    image.


    A pixel of the source image is considered to belong to the contour if
    its pixel value is equal to the input foreground value and it has in
    its neighborhood at least one pixel which its pixel value is equal to
    the input background value. The output image will have pixels which
    will be set to the output foreground value if they belong to the
    contour, otherwise they will be set to the output background value.

    The neighborhood "radius" is set thanks to the radius params.


    See:
     Image

     Neighborhood

     NeighborhoodOperator

     NeighborhoodIterator

     itk::simple::SimpleContourExtractor for the procedural interface

     itk::SimpleContourExtractorImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkSimpleContourExtractorImageFilter.h

    
### sitk.SimpleITK
### sitk.Sin
    Sin(Image image1) -> Image



    Computes the sine of each pixel.


    This function directly calls the execute method of SinImageFilter in order to support a procedural API


    See:
     itk::simple::SinImageFilter for the object oriented interface



    
### sitk.SinImageFilter


    Computes the sine of each pixel.


    The computations are performed using std::sin(x).
    See:
     itk::simple::Sin for the procedural interface

     itk::SinImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkSinImageFilter.h

    
### sitk.Slice
    Slice(Image image1, VectorInt32 start, VectorInt32 stop, VectorInt32 step) -> Image



    Slices an image based on a starting index and a stopping index, and a
    step size.


    This function directly calls the execute method of SliceImageFilter in order to support a procedural API


    See:
     itk::simple::SliceImageFilter for the object oriented interface



    
### sitk.SliceImageFilter


    Slices an image based on a starting index and a stopping index, and a
    step size.


    This class is designed to facilitate the implementation of extended
    sliced based indexing into images.

    The input and output image must be of the same dimension.

    The input parameters are a starting and stopping index as well as a
    stepping size. The starting index indicates the first pixels to be
    used and for each dimension the index is incremented by the step until
    the index is equal to or "beyond" the stopping index. If the step is
    negative then the image will be reversed in the dimension, and the
    stopping index is expected to be less then the starting index. If the
    stopping index is already beyond the starting index then an image of
    size zero will be returned.

    The output image's starting index is always zero. The origin is the
    physical location of the starting index. The output directions cosine
    matrix is that of the input but with sign changes matching that of the
    step's sign.


    In certain combinations such as with start=1, and step>1 while the
    physical location of the center of the pixel remains the same, the
    extent (edge to edge space) of the output image will be beyond the
    extent of the original image.

    See:
     itk::simple::Slice for the procedural interface

     itk::SliceImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkSliceImageFilter.h

    
### sitk.SmoothingRecursiveGaussian
    SmoothingRecursiveGaussian(Image image1, double sigma, bool normalizeAcrossScale=False) -> Image
    SmoothingRecursiveGaussian(Image image1, VectorDouble sigma, bool normalizeAcrossScale=False) -> Image



    Computes the smoothing of an image by convolution with the Gaussian
    kernels implemented as IIR filters.


    This function directly calls the execute method of SmoothingRecursiveGaussianImageFilter in order to support a procedural API


    See:
     itk::simple::SmoothingRecursiveGaussianImageFilter for the object oriented interface



    
### sitk.SmoothingRecursiveGaussianImageFilter


    Computes the smoothing of an image by convolution with the Gaussian
    kernels implemented as IIR filters.


    This filter is implemented using the recursive gaussian filters. For
    multi-component images, the filter works on each component
    independently.

    For this filter to be able to run in-place the input and output image
    types need to be the same and/or the same type as the RealImageType.
    See:
     itk::simple::SmoothingRecursiveGaussian for the procedural interface

     itk::SmoothingRecursiveGaussianImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkSmoothingRecursiveGaussianImageFilter.h

    
### sitk.SobelEdgeDetection
    SobelEdgeDetection(Image image1) -> Image



    A 2D or 3D edge detection using the Sobel operator.


    This function directly calls the execute method of SobelEdgeDetectionImageFilter in order to support a procedural API


    See:
     itk::simple::SobelEdgeDetectionImageFilter for the object oriented interface



    
### sitk.SobelEdgeDetectionImageFilter


    A 2D or 3D edge detection using the Sobel operator.


    This filter uses the Sobel operator to calculate the image gradient
    and then finds the magnitude of this gradient vector. The Sobel
    gradient magnitude (square-root sum of squares) is an indication of
    edge strength.


    See:
     ImageToImageFilter

     SobelOperator

     Neighborhood

     NeighborhoodOperator

     NeighborhoodIterator

     itk::simple::SobelEdgeDetection for the procedural interface

     itk::SobelEdgeDetectionImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkSobelEdgeDetectionImageFilter.h

    
### sitk.SpeckleNoise
    SpeckleNoise(Image image1, double standardDeviation=1.0, uint32_t seed) -> Image



    Alter an image with speckle (multiplicative) noise.


    This function directly calls the execute method of SpeckleNoiseImageFilter in order to support a procedural API


    See:
     itk::simple::SpeckleNoiseImageFilter for the object oriented interface



    
### sitk.SpeckleNoiseImageFilter


    Alter an image with speckle (multiplicative) noise.


    The speckle noise follows a gamma distribution of mean 1 and standard
    deviation provided by the user. The noise is proportional to the pixel
    intensity.

    It can be modeled as:


    $ I = I_0 \ast G $

    where $ G $ is a is a gamma distributed random variable of mean 1 and variance
    proportional to the noise level:

    $ G \sim \Gamma(\frac{1}{\sigma^2}, \sigma^2) $

    Gaetan Lehmann
     This code was contributed in the Insight Journal paper "Noise
    Simulation". https://hdl.handle.net/10380/3158
    See:
     itk::simple::SpeckleNoise for the procedural interface

     itk::SpeckleNoiseImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkSpeckleNoiseImageFilter.h

    
### sitk.Sqrt
    Sqrt(Image image1) -> Image



    Computes the square root of each pixel.


    This function directly calls the execute method of SqrtImageFilter in order to support a procedural API


    See:
     itk::simple::SqrtImageFilter for the object oriented interface



    
### sitk.SqrtImageFilter


    Computes the square root of each pixel.


    The computations are performed using std::sqrt(x).
    See:
     itk::simple::Sqrt for the procedural interface

     itk::SqrtImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkSqrtImageFilter.h

    
### sitk.Square
    Square(Image image1) -> Image



    Computes the square of the intensity values pixel-wise.


    This function directly calls the execute method of SquareImageFilter in order to support a procedural API


    See:
     itk::simple::SquareImageFilter for the object oriented interface



    
### sitk.SquareImageFilter


    Computes the square of the intensity values pixel-wise.



    See:
     itk::simple::Square for the procedural interface

     itk::SquareImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkSquareImageFilter.h

    
### sitk.SquaredDifference
    SquaredDifference(Image image1, Image image2) -> Image
    SquaredDifference(Image image1, double constant) -> Image
    SquaredDifference(double constant, Image image2) -> Image



    
### sitk.SquaredDifferenceImageFilter


    Implements pixel-wise the computation of squared difference.


    This filter is parametrized over the types of the two input images and
    the type of the output image.

    Numeric conversions (castings) are done by the C++ defaults.

    The filter will walk over all the pixels in the two input images, and
    for each one of them it will do the following:


    cast the input 1 pixel value to double

    cast the input 2 pixel value to double

    compute the difference of the two pixel values

    compute the square of the difference

    cast the double value resulting from sqr() to the pixel type of the
    output image

    store the casted value into the output image.
     The filter expect all images to have the same dimension (e.g. all 2D,
    or all 3D, or all ND)
    See:
     itk::simple::SquaredDifference for the procedural interface

     itk::SquaredDifferenceImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkSquaredDifferenceImageFilter.h

    
### sitk.StandardDeviationProjection
    StandardDeviationProjection(Image image1, unsigned int projectionDimension=0) -> Image



    Mean projection.


    This function directly calls the execute method of StandardDeviationProjectionImageFilter in order to support a procedural API


    See:
     itk::simple::StandardDeviationProjectionImageFilter for the object oriented interface



    
### sitk.StandardDeviationProjectionImageFilter


    Mean projection.


    This class was contributed to the Insight Journal by Gaetan Lehmann.
    The original paper can be found at https://hdl.handle.net/1926/164


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.

    See:
     ProjectionImageFilter

     MedianProjectionImageFilter

     MeanProjectionImageFilter

     SumProjectionImageFilter

     MeanProjectionImageFilter

     MaximumProjectionImageFilter

     MinimumProjectionImageFilter

     BinaryProjectionImageFilter

     itk::simple::StandardDeviationProjection for the procedural interface

     itk::StandardDeviationProjectionImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkStandardDeviationProjectionImageFilter.h

    
### sitk.StatisticsImageFilter


    Compute min. max, variance and mean of an Image .


    StatisticsImageFilter computes the minimum, maximum, sum, mean, variance sigma of an image.
    The filter needs all of its input image. It behaves as a filter with
    an input and output. Thus it can be inserted in a pipline with other
    filters and the statistics will only be recomputed if a downstream
    filter changes.

    The filter passes its input through unmodified. The filter is
    threaded. It computes statistics in each thread then combines them in
    its AfterThreadedGenerate method.


    See:
     itk::StatisticsImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkStatisticsImageFilter.h

    
### sitk.StochasticFractalDimension
    StochasticFractalDimension(Image image, Image maskImage, VectorUInt32 neighborhoodRadius) -> Image
    StochasticFractalDimension(Image image, VectorUInt32 neighborhoodRadius) -> Image



    
### sitk.StochasticFractalDimensionImageFilter


    This filter computes the stochastic fractal dimension of the input
    image.


    The methodology is based on Madelbrot's fractal theory and the concept
    of fractional Brownian motion and yields images which have been used
    for classification and edge enhancement.

    This class which is templated over the input and output images as well
    as a mask image type. The input is a scalar image, an optional
    neighborhood radius (default = 2), and an optional mask. The mask can
    be specified to decrease computation time since, as the authors point
    out, calculation is time-consuming.

    This filter was contributed by Nick Tustison and James Gee from the
    PICSL lab, at the University of Pennsylvania as an paper to the
    Insight Journal:

    "Stochastic Fractal Dimension Image" https://hdl.handle.net/1926/1525 http://www.insight-journal.org/browse/publication/318


    Nick Tustison

    See:
     itk::simple::StochasticFractalDimension for the procedural interface

     itk::StochasticFractalDimensionImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkStochasticFractalDimensionImageFilter.h

    
### sitk.Subtract
    Subtract(Image image1, Image image2) -> Image
    Subtract(Image image1, double constant) -> Image
    Subtract(double constant, Image image2) -> Image



    
### sitk.SubtractImageFilter


    Pixel-wise subtraction of two images.


    Subtract each pixel from image2 from its corresponding pixel in
    image1:


    This is done using


    This class is templated over the types of the two input images and the
    type of the output image. Numeric conversions (castings) are done by
    the C++ defaults.

    Additionally, a constant can be subtracted from every pixel in an
    image using:



    The result of AddImageFilter with a negative constant is not necessarily the same as SubtractImageFilter . This would be the case when the PixelType defines an operator-() that is not the inverse of operator+()

    See:
     itk::simple::Subtract for the procedural interface

     itk::SubtractImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkSubtractImageFilter.h

    
### sitk.SumProjection
    SumProjection(Image image1, unsigned int projectionDimension=0) -> Image



    Sum projection.


    This function directly calls the execute method of SumProjectionImageFilter in order to support a procedural API


    See:
     itk::simple::SumProjectionImageFilter for the object oriented interface



    
### sitk.SumProjectionImageFilter


    Sum projection.


    This class was contributed to the Insight Journal by Gaetan Lehmann.
    The original paper can be found at https://hdl.handle.net/1926/164


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.

    See:
     ProjectionImageFilter

     MedianProjectionImageFilter

     MeanProjectionImageFilter

     MeanProjectionImageFilter

     MaximumProjectionImageFilter

     MinimumProjectionImageFilter

     BinaryProjectionImageFilter

     StandardDeviationProjectionImageFilter

     itk::simple::SumProjection for the procedural interface

     itk::SumProjectionImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkSumProjectionImageFilter.h

    
### sitk.SwigPyIteratorProxy of C++ swig::SwigPyIterator class.
### sitk.SymmetricForcesDemonsRegistrationFilter


    Deformably register two images using the demons algorithm.


    This class was contributed by Corinne Mattmann, ETH Zurich,
    Switzerland. based on a variation of the DemonsRegistrationFilter . The basic modification is to use equation (5) from Thirion's paper
    along with the modification for avoiding large deformations when
    gradients have small values.

    SymmetricForcesDemonsRegistrationFilter implements the demons deformable algorithm that register two images
    by computing the deformation field which will map a moving image onto
    a fixed image.

    A deformation field is represented as a image whose pixel type is some
    vector type with at least N elements, where N is the dimension of the
    fixed image. The vector type must support element access via operator
    []. It is assumed that the vector elements behave like floating point
    scalars.

    This class is templated over the fixed image type, moving image type
    and the deformation field type.

    The input fixed and moving images are set via methods SetFixedImage
    and SetMovingImage respectively. An initial deformation field maybe
    set via SetInitialDisplacementField or SetInput. If no initial field
    is set, a zero field is used as the initial condition.

    The algorithm has one parameters: the number of iteration to be
    performed.

    The output deformation field can be obtained via methods GetOutput or
    GetDisplacementField.

    This class make use of the finite difference solver hierarchy. Update
    for each iteration is computed in DemonsRegistrationFunction .


    WARNING:
    This filter assumes that the fixed image type, moving image type and
    deformation field type all have the same number of dimensions.

    See:
     SymmetricForcesDemonsRegistrationFunction

     DemonsRegistrationFilter

     DemonsRegistrationFunction

     itk::SymmetricForcesDemonsRegistrationFilter for the Doxygen on the original ITK class.


    C++ includes: sitkSymmetricForcesDemonsRegistrationFilter.h

    
### sitk.Tan
    Tan(Image image1) -> Image



    Computes the tangent of each input pixel.


    This function directly calls the execute method of TanImageFilter in order to support a procedural API


    See:
     itk::simple::TanImageFilter for the object oriented interface



    
### sitk.TanImageFilter


    Computes the tangent of each input pixel.


    The computations are performed using std::tan(x).
    See:
     itk::simple::Tan for the procedural interface

     itk::TanImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkTanImageFilter.h

    
### sitk.TernaryAdd
    TernaryAdd(Image image1, Image image2, Image image3) -> Image



    Pixel-wise addition of three images.


    This function directly calls the execute method of TernaryAddImageFilter in order to support a procedural API


    See:
     itk::simple::TernaryAddImageFilter for the object oriented interface



    
### sitk.TernaryAddImageFilter


    Pixel-wise addition of three images.


    This class is templated over the types of the three input images and
    the type of the output image. Numeric conversions (castings) are done
    by the C++ defaults.
    See:
     itk::simple::TernaryAdd for the procedural interface

     itk::TernaryAddImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkTernaryAddImageFilter.h

    
### sitk.TernaryMagnitude
    TernaryMagnitude(Image image1, Image image2, Image image3) -> Image



    Compute the pixel-wise magnitude of three images.


    This function directly calls the execute method of TernaryMagnitudeImageFilter in order to support a procedural API


    See:
     itk::simple::TernaryMagnitudeImageFilter for the object oriented interface



    
### sitk.TernaryMagnitudeImageFilter


    Compute the pixel-wise magnitude of three images.


    This class is templated over the types of the three input images and
    the type of the output image. Numeric conversions (castings) are done
    by the C++ defaults.
    See:
     itk::simple::TernaryMagnitude for the procedural interface

     itk::TernaryMagnitudeImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkTernaryMagnitudeImageFilter.h

    
### sitk.TernaryMagnitudeSquared
    TernaryMagnitudeSquared(Image image1, Image image2, Image image3) -> Image



    Compute the pixel-wise squared magnitude of three images.


    This function directly calls the execute method of TernaryMagnitudeSquaredImageFilter in order to support a procedural API


    See:
     itk::simple::TernaryMagnitudeSquaredImageFilter for the object oriented interface



    
### sitk.TernaryMagnitudeSquaredImageFilter


    Compute the pixel-wise squared magnitude of three images.


    This class is templated over the types of the three input images and
    the type of the output image. Numeric conversions (castings) are done
    by the C++ defaults.
    See:
     itk::simple::TernaryMagnitudeSquared for the procedural interface

     itk::TernaryMagnitudeSquaredImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkTernaryMagnitudeSquaredImageFilter.h

    
### sitk.Threshold
    Threshold(Image image1, double lower=0.0, double upper=1.0, double outsideValue=0.0) -> Image



    Set image values to a user-specified value if they are below, above,
    or between simple threshold values.


    This function directly calls the execute method of ThresholdImageFilter in order to support a procedural API


    See:
     itk::simple::ThresholdImageFilter for the object oriented interface



    
### sitk.ThresholdImageFilter


    Set image values to a user-specified value if they are below, above,
    or between simple threshold values.


    ThresholdImageFilter sets image values to a user-specified "outside" value (by default,
    "black") if the image values are below, above, or between simple
    threshold values.

    The available methods are:

    ThresholdAbove() : The values greater than the threshold value are set
    to OutsideValue

    ThresholdBelow() : The values less than the threshold value are set to
    OutsideValue

    ThresholdOutside() : The values outside the threshold range (less than
    lower or greater than upper) are set to OutsideValue

    Note that these definitions indicate that pixels equal to the
    threshold value are not set to OutsideValue in any of these methods

    The pixels must support the operators >= and <=.
    See:
     itk::simple::Threshold for the procedural interface

     itk::ThresholdImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkThresholdImageFilter.h

    
### sitk.ThresholdMaximumConnectedComponents
    ThresholdMaximumConnectedComponents(Image image1, uint32_t minimumObjectSizeInPixels=0, double upperBoundary, uint8_t insideValue=1, uint8_t outsideValue=0) -> Image



    Finds the threshold value of an image based on maximizing the number
    of objects in the image that are larger than a given minimal size.


    This function directly calls the execute method of ThresholdMaximumConnectedComponentsImageFilter in order to support a procedural API


    See:
     itk::simple::ThresholdMaximumConnectedComponentsImageFilter for the object oriented interface



    
### sitk.ThresholdMaximumConnectedComponentsImageFilter


    Finds the threshold value of an image based on maximizing the number
    of objects in the image that are larger than a given minimal size.



    This method is based on Topological Stable State Thresholding to
    calculate the threshold set point. This method is particularly
    effective when there are a large number of objects in a microscopy
    image. Compiling in Debug mode and enable the debug flag for this
    filter to print debug information to see how the filter focuses in on
    a threshold value. Please see the Insight Journal's MICCAI 2005
    workshop for a complete description. References are below.
    Parameters
    The MinimumObjectSizeInPixels parameter is controlled through the
    class Get/SetMinimumObjectSizeInPixels() method. Similar to the
    standard itk::BinaryThresholdImageFilter the Get/SetInside and Get/SetOutside values of the threshold can be
    set. The GetNumberOfObjects() and GetThresholdValue() methods return
    the number of objects above the minimum pixel size and the calculated
    threshold value.
    Automatic Thresholding in ITK
    There are multiple methods to automatically calculate the threshold
    intensity value of an image. As of version 4.0, ITK has a Thresholding
    ( ITKThresholding ) module which contains numerous automatic
    thresholding methods.implements two of these. Topological Stable State
    Thresholding works well on images with a large number of objects to be
    counted.
    References:
    1) Urish KL, August J, Huard J. "Unsupervised segmentation for
    myofiber counting in immunoflourescent images". Insight Journal. ISC
    /NA-MIC/MICCAI Workshop on Open-Source Software (2005) Dspace handle: https://hdl.handle.net/1926/48 2) Pikaz A, Averbuch, A. "Digital image thresholding based on
    topological stable-state". Pattern Recognition, 29(5): 829-843, 1996.

    Questions: email Ken Urish at ken.urish(at)gmail.com Please cc the itk
    list serve for archival purposes.

    See:
     itk::simple::ThresholdMaximumConnectedComponents for the procedural interface

     itk::ThresholdMaximumConnectedComponentsImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkThresholdMaximumConnectedComponentsImageFilter.h

    
### sitk.ThresholdSegmentationLevelSet
    ThresholdSegmentationLevelSet(Image image1, Image image2, double lowerThreshold=0.0, double upperThreshold=255.0, double maximumRMSError=0.02, double propagationScaling=1.0, double curvatureScaling=1.0, uint32_t numberOfIterations=1000, bool reverseExpansionDirection=False) -> Image



    Segments structures in images based on intensity values.


    This function directly calls the execute method of ThresholdSegmentationLevelSetImageFilter in order to support a procedural API


    See:
     itk::simple::ThresholdSegmentationLevelSetImageFilter for the object oriented interface



    
### sitk.ThresholdSegmentationLevelSetImageFilter


    Segments structures in images based on intensity values.


    IMPORTANT
    The SegmentationLevelSetImageFilter class and the ThresholdSegmentationLevelSetFunction class contain additional information necessary to the full
    understanding of how to use this filter.
    OVERVIEW
    This class is a level set method segmentation filter. It constructs a
    speed function which is close to zero at the upper and lower bounds of
    an intensity window, effectively locking the propagating front onto
    those edges. Elsewhere, the front will propagate quickly.
    INPUTS
    This filter requires two inputs. The first input is a seed image. This
    seed image must contain an isosurface that you want to use as the seed
    for your segmentation. It can be a binary, graylevel, or floating
    point image. The only requirement is that it contain a closed
    isosurface that you will identify as the seed by setting the
    IsosurfaceValue parameter of the filter. For a binary image you will
    want to set your isosurface value halfway between your on and off
    values (i.e. for 0's and 1's, use an isosurface value of 0.5).

    The second input is the feature image. This is the image from which
    the speed function will be calculated. For most applications, this is
    the image that you want to segment. The desired isosurface in your
    seed image should lie within the region of your feature image that you
    are trying to segment. Note that this filter does no preprocessing of
    the feature image before thresholding.

    See SegmentationLevelSetImageFilter for more information on Inputs.
    OUTPUTS
    The filter outputs a single, scalar, real-valued image. Positive
    values in the output image are inside the segmentated region and
    negative values in the image are outside of the inside region. The
    zero crossings of the image correspond to the position of the level
    set front.

    See SparseFieldLevelSetImageFilter and SegmentationLevelSetImageFilter for more information.
    PARAMETERS
    In addition to parameters described in SegmentationLevelSetImageFilter , this filter adds the UpperThreshold and LowerThreshold. See ThresholdSegmentationLevelSetFunction for a description of how these values affect the segmentation.

    See:
     SegmentationLevelSetImageFilter

     ThresholdSegmentationLevelSetFunction ,

     SparseFieldLevelSetImageFilter

     itk::simple::ThresholdSegmentationLevelSet for the procedural interface

     itk::ThresholdSegmentationLevelSetImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkThresholdSegmentationLevelSetImageFilter.h

    
### sitk.TikhonovDeconvolution
    TikhonovDeconvolution(Image image1, Image image2, double regularizationConstant=0.0, bool normalize=False, itk::simple::TikhonovDeconvolutionImageFilter::BoundaryConditionType boundaryCondition, itk::simple::TikhonovDeconvolutionImageFilter::OutputRegionModeType outputRegionMode) -> Image



    An inverse deconvolution filter regularized in the Tikhonov sense.


    This function directly calls the execute method of TikhonovDeconvolutionImageFilter in order to support a procedural API


    See:
     itk::simple::TikhonovDeconvolutionImageFilter for the object oriented interface



    
### sitk.TikhonovDeconvolutionImageFilter


    An inverse deconvolution filter regularized in the Tikhonov sense.


    The Tikhonov deconvolution filter is the inverse deconvolution filter
    with a regularization term added to the denominator. The filter
    minimizes the equation \[ ||\hat{f} \otimes h - g||_{L_2}^2 + \mu||\hat{f}||^2
    \] where $\hat{f}$ is the estimate of the unblurred image, $h$ is the blurring kernel, $g$ is the blurred image, and $\mu$ is a non-negative real regularization function.

    The filter applies a kernel described in the Fourier domain as $H^*(\omega) / (|H(\omega)|^2 + \mu)$ where $H(\omega)$ is the Fourier transform of $h$ . The term $\mu$ is called RegularizationConstant in this filter. If $\mu$ is set to zero, this filter is equivalent to the InverseDeconvolutionImageFilter .


    Gaetan Lehmann, Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France

    Cory Quammen, The University of North Carolina at Chapel Hill

    See:
     itk::simple::TikhonovDeconvolution for the procedural interface

     itk::TikhonovDeconvolutionImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkTikhonovDeconvolutionImageFilter.h

    
### sitk.Tile
    Tile(VectorOfImage images, VectorUInt32 layout, double defaultPixelValue=0.0) -> Image
    Tile(Image image1, VectorUInt32 layout, double defaultPixelValue=0.0) -> Image
    Tile(Image image1, Image image2, VectorUInt32 layout, double defaultPixelValue=0.0) -> Image
    Tile(Image image1, Image image2, Image image3, VectorUInt32 layout, double defaultPixelValue=0.0) -> Image
    Tile(Image image1, Image image2, Image image3, Image image4, VectorUInt32 layout, double defaultPixelValue=0.0) -> Image
    Tile(Image image1, Image image2, Image image3, Image image4, Image image5, VectorUInt32 layout, double defaultPixelValue=0.0) -> Image
    
### sitk.TileImageFilter


    Tile multiple input images into a single output image.


    This filter will tile multiple images using a user-specified layout.
    The tile sizes will be large enough to accommodate the largest image
    for each tile. The layout is specified with the SetLayout method. The
    layout has the same dimension as the output image. If all entries of
    the layout are positive, the tiled output will contain the exact
    number of tiles. If the layout contains a 0 in the last dimension, the
    filter will compute a size that will accommodate all of the images.
    Empty tiles are filled with the value specified with the SetDefault
    value method. The input images must have a dimension less than or
    equal to the output image. The output image have a larger dimension
    than the input images. This filter can be used to create a volume from
    a series of inputs by specifying a layout of 1,1,0.


    See:
     itk::simple::Tile for the procedural interface


    C++ includes: sitkTileImageFilter.h

    
### sitk.Toboggan
    Toboggan(Image image1) -> Image



    toboggan image segmentation The Toboggan segmentation takes a gradient
    magnitude image as input and produces an (over-)segmentation of the
    image based on connecting each pixel to a local minimum of gradient.
    It is roughly equivalent to a watershed segmentation of the lowest
    level.


    This function directly calls the execute method of TobogganImageFilter in order to support a procedural API


    See:
     itk::simple::TobogganImageFilter for the object oriented interface



    
### sitk.TobogganImageFilter


    toboggan image segmentation The Toboggan segmentation takes a gradient
    magnitude image as input and produces an (over-)segmentation of the
    image based on connecting each pixel to a local minimum of gradient.
    It is roughly equivalent to a watershed segmentation of the lowest
    level.


    The output is a 4 connected labeled map of the image.
    See:
     itk::simple::Toboggan for the procedural interface

     itk::TobogganImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkTobogganImageFilter.h

    
### sitk.Transform


    A simplified wrapper around a variety of ITK transforms.


    The interface to ITK transform objects to be used with the ImageRegistrationMethod, ResampleImageFilter and other SimpleITK process objects. The transforms are designed to
    have a serialized array of parameters to facilitate optimization for
    registration.

    Provides a base class interface to any type of ITK transform. Objects
    of this type may have their interface converted to a derived interface
    while keeping the same reference to the ITK object.

    Additionally, this class provides a basic interface to a composite
    transforms.


    See:
     itk::CompositeTransform


    C++ includes: sitkTransform.h

    
### sitk.TransformToDisplacementField
    TransformToDisplacementField(Transform transform, itk::simple::PixelIDValueEnum outputPixelType, VectorUInt32 size, VectorDouble outputOrigin, VectorDouble outputSpacing, VectorDouble outputDirection) -> Image



    Generate a displacement field from a coordinate transform.


    This function directly calls the execute method of TransformToDisplacementFieldFilter in order to support a procedural API


    See:
     itk::simple::TransformToDisplacementFieldFilter for the object oriented interface



    
### sitk.TransformToDisplacementFieldFilter


    Generate a displacement field from a coordinate transform.


    Output information (spacing, size and direction) for the output image
    should be set. This information has the normal defaults of unit
    spacing, zero origin and identity direction. Optionally, the output
    information can be obtained from a reference image. If the reference
    image is provided and UseReferenceImage is On, then the spacing,
    origin and direction of the reference image will be used.

    Since this filter produces an image which is a different size than its
    input, it needs to override several of the methods defined in ProcessObject in order to properly manage the pipeline execution model. In
    particular, this filter overrides
    ProcessObject::GenerateOutputInformation() .

    This filter is implemented as a multithreaded filter. It provides a
    ThreadedGenerateData() method for its implementation.


    Marius Staring, Leiden University Medical Center, The Netherlands.
     This class was taken from the Insight Journal paper: https://hdl.handle.net/1926/1387
    See:
     itk::simple::TransformToDisplacementFieldFilter for the procedural interface

     itk::TransformToDisplacementFieldFilter for the Doxygen on the original ITK class.


    C++ includes: sitkTransformToDisplacementFieldFilter.h

    
### sitk.TranslationTransform


    Translation of a 2D or 3D coordinate space.



    See:
     itk::TranslationTransform


    C++ includes: sitkTranslationTransform.h

    
### sitk.TriangleThreshold
    TriangleThreshold(Image image, Image maskImage, uint8_t insideValue=1, uint8_t outsideValue=0, uint32_t numberOfHistogramBins=256, bool maskOutput=True, uint8_t maskValue=255) -> Image
    TriangleThreshold(Image image, uint8_t insideValue=1, uint8_t outsideValue=0, uint32_t numberOfHistogramBins=256, bool maskOutput=True, uint8_t maskValue=255) -> Image



    
### sitk.TriangleThresholdImageFilter


    Threshold an image using the Triangle Threshold.


    This filter creates a binary thresholded image that separates an image
    into foreground and background components. The filter computes the
    threshold using the TriangleThresholdCalculator and applies that threshold to the input image using the BinaryThresholdImageFilter .


    Richard Beare. Department of Medicine, Monash University, Melbourne,
    Australia.

    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.
     This implementation was taken from the Insight Journal paper: https://hdl.handle.net/10380/3279 or http://www.insight-journal.org/browse/publication/811


    See:
     HistogramThresholdImageFilter

     itk::simple::TriangleThreshold for the procedural interface

     itk::TriangleThresholdImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkTriangleThresholdImageFilter.h

    
### sitk.UnaryMinus
    UnaryMinus(Image image1) -> Image



    Implements pixel-wise generic operation on one image.


    This function directly calls the execute method of UnaryMinusImageFilter in order to support a procedural API


    See:
     itk::simple::UnaryMinusImageFilter for the object oriented interface



    
### sitk.UnaryMinusImageFilter


    Implements pixel-wise generic operation on one image.


    This class is parameterized over the type of the input image and the
    type of the output image. It is also parameterized by the operation to
    be applied, using a Functor style.

    UnaryFunctorImageFilter allows the output dimension of the filter to be larger than the input
    dimension. Thus subclasses of the UnaryFunctorImageFilter (like the CastImageFilter ) can be used to promote a 2D image to a 3D image, etc.


    See:
     BinaryFunctorImageFilter TernaryFunctorImageFilter

     itk::simple::UnaryMinus for the procedural interface

     itk::UnaryFunctorImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkUnaryMinusImageFilter.h

    
### sitk.UnsharpMask
    UnsharpMask(Image image1, VectorDouble sigmas, double amount=0.5, double threshold=0.0) -> Image



    Edge enhancement filter.


    This function directly calls the execute method of UnsharpMaskImageFilter in order to support a procedural API


    See:
     itk::simple::UnsharpMaskImageFilter for the object oriented interface



    
### sitk.UnsharpMaskImageFilter


    Edge enhancement filter.


    This filter subtracts a smoothed version of the image from the image
    to achieve the edge enhancing effect. https://en.wikipedia.org/w/index.php?title=Unsharp_masking&oldid=75048
    6803#Photographic_unsharp_masking

    It has configurable amount, radius (sigma) and threshold, and whether
    to clamp the resulting values to the range of output type.

    Formula: sharpened=original+[abs(original-blurred)-threshold]*amount

    If clamping is turned off (it is on by default), casting to output
    pixel format is done using C++ defaults, meaning that values are not
    clamped but rather wrap around e.g. 260 -> 4 (unsigned char).


    See:
     ImageToImageFilter

     SmoothingRecursiveGaussianImageFilter

     RescaleIntensityImageFilter

     itk::simple::UnsharpMask for the procedural interface

     itk::UnsharpMaskImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkUnsharpMaskImageFilter.h

    
### sitk.ValuedRegionalMaxima
    ValuedRegionalMaxima(Image image1, bool fullyConnected=False) -> Image



    Transforms the image so that any pixel that is not a regional maxima
    is set to the minimum value for the pixel type. Pixels that are
    regional maxima retain their value.


    This function directly calls the execute method of ValuedRegionalMaximaImageFilter in order to support a procedural API


    See:
     itk::simple::ValuedRegionalMaximaImageFilter for the object oriented interface



    
### sitk.ValuedRegionalMaximaImageFilter


    Transforms the image so that any pixel that is not a regional maxima
    is set to the minimum value for the pixel type. Pixels that are
    regional maxima retain their value.


    Regional maxima are flat zones surrounded by pixels of lower value. A
    completely flat image will be marked as a regional maxima by this
    filter.

    This code was contributed in the Insight Journal paper: "Finding
    regional extrema - methods and performance" by Beare R., Lehmann G. https://hdl.handle.net/1926/153 http://www.insight-journal.org/browse/publication/65


    Richard Beare. Department of Medicine, Monash University, Melbourne,
    Australia.

    See:
     ValuedRegionalMinimaImageFilter

     ValuedRegionalExtremaImageFilter

     HMinimaImageFilter

     itk::simple::ValuedRegionalMaxima for the procedural interface

     itk::ValuedRegionalMaximaImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkValuedRegionalMaximaImageFilter.h

    
### sitk.ValuedRegionalMinima
    ValuedRegionalMinima(Image image1, bool fullyConnected=False) -> Image



    Transforms the image so that any pixel that is not a regional minima
    is set to the maximum value for the pixel type. Pixels that are
    regional minima retain their value.


    This function directly calls the execute method of ValuedRegionalMinimaImageFilter in order to support a procedural API


    See:
     itk::simple::ValuedRegionalMinimaImageFilter for the object oriented interface



    
### sitk.ValuedRegionalMinimaImageFilter


    Transforms the image so that any pixel that is not a regional minima
    is set to the maximum value for the pixel type. Pixels that are
    regional minima retain their value.


    Regional minima are flat zones surrounded by pixels of higher value. A
    completely flat image will be marked as a regional minima by this
    filter.

    This code was contributed in the Insight Journal paper: "Finding
    regional extrema - methods and performance" by Beare R., Lehmann G. https://hdl.handle.net/1926/153 http://www.insight-journal.org/browse/publication/65


    Richard Beare. Department of Medicine, Monash University, Melbourne,
    Australia.

    See:
     ValuedRegionalMaximaImageFilter , ValuedRegionalExtremaImageFilter ,

     HMinimaImageFilter

     itk::simple::ValuedRegionalMinima for the procedural interface

     itk::ValuedRegionalMinimaImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkValuedRegionalMinimaImageFilter.h

    
### sitk.VectorBoolProxy of C++ std::vector<(bool)> class.
### sitk.VectorConfidenceConnected
    VectorConfidenceConnected(Image image1, VectorUIntList seedList, unsigned int numberOfIterations=4, double multiplier=4.5, unsigned int initialNeighborhoodRadius=1, uint8_t replaceValue=1) -> Image



    itk::simple::VectorConfidenceConnectedImageFilter Functional Interface

    This function directly calls the execute method of VectorConfidenceConnectedImageFilter in order to support a fully functional API


    
### sitk.VectorConfidenceConnectedImageFilter


    Segment pixels with similar statistics using connectivity.


    This filter extracts a connected set of pixels whose pixel intensities
    are consistent with the pixel statistics of a seed point. The mean and
    variance across a neighborhood (8-connected, 26-connected, etc.) are
    calculated for a seed point. Then pixels connected to this seed point
    whose values are within the confidence interval for the seed point are
    grouped. The width of the confidence interval is controlled by the
    "Multiplier" variable (the confidence interval is the mean plus or
    minus the "Multiplier" times the standard deviation). If the
    intensity variations across a segment were gaussian, a "Multiplier"
    setting of 2.5 would define a confidence interval wide enough to
    capture 99% of samples in the segment.

    After this initial segmentation is calculated, the mean and variance
    are re-calculated. All the pixels in the previous segmentation are
    used to calculate the mean the standard deviation (as opposed to using
    the pixels in the neighborhood of the seed point). The segmentation is
    then recalculted using these refined estimates for the mean and
    variance of the pixel values. This process is repeated for the
    specified number of iterations. Setting the "NumberOfIterations" to
    zero stops the algorithm after the initial segmentation from the seed
    point.

    NOTE: the lower and upper threshold are restricted to lie within the
    valid numeric limits of the input data pixel type. Also, the limits
    may be adjusted to contain the seed point's intensity.
    See:
     itk::simple::VectorConfidenceConnected for the procedural interface

     itk::VectorConfidenceConnectedImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkVectorConfidenceConnectedImageFilter.h

    
### sitk.VectorConnectedComponent
    VectorConnectedComponent(Image image1, double distanceThreshold=1.0, bool fullyConnected=False) -> Image



    A connected components filter that labels the objects in a vector
    image. Two vectors are pointing similar directions if one minus their
    dot product is less than a threshold. Vectors that are 180 degrees out
    of phase are similar. Assumes that vectors are normalized.


    This function directly calls the execute method of VectorConnectedComponentImageFilter in order to support a procedural API


    See:
     itk::simple::VectorConnectedComponentImageFilter for the object oriented interface



    
### sitk.VectorConnectedComponentImageFilter


    A connected components filter that labels the objects in a vector
    image. Two vectors are pointing similar directions if one minus their
    dot product is less than a threshold. Vectors that are 180 degrees out
    of phase are similar. Assumes that vectors are normalized.



    See:
     itk::simple::VectorConnectedComponent for the procedural interface

     itk::VectorConnectedComponentImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkVectorConnectedComponentImageFilter.h

    
### sitk.VectorDoubleProxy of C++ std::vector<(double)> class.
### sitk.VectorFloatProxy of C++ std::vector<(float)> class.
### sitk.VectorIndexSelectionCast
    VectorIndexSelectionCast(Image image1, unsigned int index=0, itk::simple::PixelIDValueEnum outputPixelType) -> Image



    Extracts the selected index of the vector that is the input pixel
    type.


    This function directly calls the execute method of VectorIndexSelectionCastImageFilter in order to support a procedural API


    See:
     itk::simple::VectorIndexSelectionCastImageFilter for the object oriented interface



    
### sitk.VectorIndexSelectionCastImageFilter


    Extracts the selected index of the vector that is the input pixel
    type.


    This filter is templated over the input image type and output image
    type.

    The filter expect the input image pixel type to be a vector and the
    output image pixel type to be a scalar. The only requirement on the
    type used for representing the vector is that it must provide an
    operator[].


    See:
     ComposeImageFilter

     itk::simple::VectorIndexSelectionCast for the procedural interface

     itk::VectorIndexSelectionCastImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkVectorIndexSelectionCastImageFilter.h

    
### sitk.VectorInt16Proxy of C++ std::vector<(int16_t)> class.
### sitk.VectorInt32Proxy of C++ std::vector<(int32_t)> class.
### sitk.VectorInt64Proxy of C++ std::vector<(int64_t)> class.
### sitk.VectorInt8Proxy of C++ std::vector<(int8_t)> class.
### sitk.VectorMagnitude
    VectorMagnitude(Image image1) -> Image



    Take an image of vectors as input and produce an image with the
    magnitude of those vectors.


    This function directly calls the execute method of VectorMagnitudeImageFilter in order to support a procedural API


    See:
     itk::simple::VectorMagnitudeImageFilter for the object oriented interface



    
### sitk.VectorMagnitudeImageFilter


    Take an image of vectors as input and produce an image with the
    magnitude of those vectors.


    The filter expects the input image pixel type to be a vector and the
    output image pixel type to be a scalar.

    This filter assumes that the PixelType of the input image is a
    VectorType that provides a GetNorm() method.
    See:
     itk::simple::VectorMagnitude for the procedural interface

     itk::VectorMagnitudeImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkVectorMagnitudeImageFilter.h

    
### sitk.VectorOfImageProxy of C++ std::vector<(itk::simple::Image)> class.
### sitk.VectorStringProxy of C++ std::vector<(std::string)> class.
### sitk.VectorUInt16Proxy of C++ std::vector<(uint16_t)> class.
### sitk.VectorUInt32Proxy of C++ std::vector<(uint32_t)> class.
### sitk.VectorUInt64Proxy of C++ std::vector<(uint64_t)> class.
### sitk.VectorUInt8Proxy of C++ std::vector<(uint8_t)> class.
### sitk.VectorUIntListProxy of C++ std::vector<(std::vector<(unsigned int)>)> class.
### sitk.Version


    Version info for SimpleITK.

    C++ includes: sitkVersion.h

    
### sitk.Version_BuildDateVersion_BuildDate() -> std::string const &
### sitk.Version_ExtendedVersionStringVersion_ExtendedVersionString() -> std::string const &
### sitk.Version_ITKMajorVersionVersion_ITKMajorVersion() -> unsigned int
### sitk.Version_ITKMinorVersionVersion_ITKMinorVersion() -> unsigned int
### sitk.Version_ITKModulesEnabledVersion_ITKModulesEnabled() -> VectorString
### sitk.Version_ITKPatchVersionVersion_ITKPatchVersion() -> unsigned int
### sitk.Version_ITKVersionStringVersion_ITKVersionString() -> std::string const &
### sitk.Version_MajorVersionVersion_MajorVersion() -> unsigned int
### sitk.Version_MinorVersionVersion_MinorVersion() -> unsigned int
### sitk.Version_PatchVersionVersion_PatchVersion() -> unsigned int
### sitk.Version_TweakVersionVersion_TweakVersion() -> unsigned int
### sitk.Version_VersionStringVersion_VersionString() -> std::string const &
### sitk.VersorRigid3DTransform


    A rotation as a versor around a fixed center with translation of a 3D
    coordinate space.



    See:
     itk::VersorRigid3DTransform


    C++ includes: sitkVersorRigid3DTransform.h

    
### sitk.VersorTransform


    A 3D rotation transform with rotation as a versor around a fixed
    center.



    See:
     itk::VersorTransform


    C++ includes: sitkVersorTransform.h

    
### sitk.VotingBinary
    VotingBinary(Image image1, VectorUInt32 radius, unsigned int birthThreshold=1, unsigned int survivalThreshold=1, double foregroundValue=1.0, double backgroundValue=0.0) -> Image



    Applies a voting operation in a neighborhood of each pixel.


    This function directly calls the execute method of VotingBinaryImageFilter in order to support a procedural API


    See:
     itk::simple::VotingBinaryImageFilter for the object oriented interface



    
### sitk.VotingBinaryHoleFilling
    VotingBinaryHoleFilling(Image image1, VectorUInt32 radius, unsigned int majorityThreshold=1, double foregroundValue=1.0, double backgroundValue=0.0) -> Image



    Fills in holes and cavities by applying a voting operation on each
    pixel.


    This function directly calls the execute method of VotingBinaryHoleFillingImageFilter in order to support a procedural API


    See:
     itk::simple::VotingBinaryHoleFillingImageFilter for the object oriented interface



    
### sitk.VotingBinaryHoleFillingImageFilter


    Fills in holes and cavities by applying a voting operation on each
    pixel.



    See:
     Image

     VotingBinaryImageFilter

     VotingBinaryIterativeHoleFillingImageFilter

     Neighborhood

     NeighborhoodOperator

     NeighborhoodIterator

     itk::simple::VotingBinaryHoleFilling for the procedural interface

     itk::VotingBinaryHoleFillingImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkVotingBinaryHoleFillingImageFilter.h

    
### sitk.VotingBinaryImageFilter


    Applies a voting operation in a neighborhood of each pixel.



    Pixels which are not Foreground or Background will remain unchanged.

    See:
     Image

     Neighborhood

     NeighborhoodOperator

     NeighborhoodIterator

     itk::simple::VotingBinary for the procedural interface

     itk::VotingBinaryImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkVotingBinaryImageFilter.h

    
### sitk.VotingBinaryIterativeHoleFilling
    VotingBinaryIterativeHoleFilling(Image image1, VectorUInt32 radius, unsigned int maximumNumberOfIterations=10, unsigned int majorityThreshold=1, double foregroundValue=1.0, double backgroundValue=0.0) -> Image



    Fills in holes and cavities by iteratively applying a voting
    operation.


    This function directly calls the execute method of VotingBinaryIterativeHoleFillingImageFilter in order to support a procedural API


    See:
     itk::simple::VotingBinaryIterativeHoleFillingImageFilter for the object oriented interface



    
### sitk.VotingBinaryIterativeHoleFillingImageFilter


    Fills in holes and cavities by iteratively applying a voting
    operation.


    This filter uses internally the VotingBinaryHoleFillingImageFilter , and runs it iteratively until no pixels are being changed or until
    it reaches the maximum number of iterations. The purpose of the filter
    is to fill in holes of medium size (tens of pixels in radius). In
    principle the number of iterations is related to the size of the holes
    to be filled in. The larger the holes, the more iteration must be run
    with this filter in order to fill in the full hole. The size of the
    neighborhood is also related to the curvature of the hole borders and
    therefore the hole size. Note that as a collateral effect this filter
    may also fill in cavities in the external side of structures.

    This filter is templated over a single image type because the output
    image type must be the same as the input image type. This is required
    in order to make the iterations possible, since the output image of
    one iteration is taken as the input image for the next iteration.


    See:
     Image

     VotingBinaryImageFilter

     VotingBinaryHoleFillingImageFilter

     Neighborhood

     NeighborhoodOperator

     NeighborhoodIterator

     itk::simple::VotingBinaryIterativeHoleFilling for the procedural interface

     itk::VotingBinaryIterativeHoleFillingImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkVotingBinaryIterativeHoleFillingImageFilter.h

    
### sitk.Warp
    Warp(Image image, Image displacementField, itk::simple::InterpolatorEnum interpolator, VectorUInt32 outputSize, VectorDouble outputOrigin, VectorDouble outputSpacing, VectorDouble outputDirection, double edgePaddingValue=0.0) -> Image



    Warps an image using an input displacement field.


    This function directly calls the execute method of WarpImageFilter in order to support a procedural API


    See:
     itk::simple::WarpImageFilter for the object oriented interface



    
### sitk.WarpImageFilter


    Warps an image using an input displacement field.


    WarpImageFilter warps an existing image with respect to a given displacement field.

    A displacement field is represented as a image whose pixel type is
    some vector type with at least N elements, where N is the dimension of
    the input image. The vector type must support element access via
    operator [].

    The output image is produced by inverse mapping: the output pixels are
    mapped back onto the input image. This scheme avoids the creation of
    any holes and overlaps in the output image.

    Each vector in the displacement field represent the distance between a
    geometric point in the input space and a point in the output space
    such that:

    \[ p_{in} = p_{out} + d \]

    Typically the mapped position does not correspond to an integer pixel
    position in the input image. Interpolation via an image function is
    used to compute values at non-integer positions. The default
    interpolation typed used is the LinearInterpolateImageFunction . The user can specify a particular interpolation function via SetInterpolator() . Note that the input interpolator must derive from base class InterpolateImageFunction .

    Position mapped to outside of the input image buffer are assigned a
    edge padding value.

    The LargetPossibleRegion for the output is inherited from the input
    displacement field. The output image spacing, origin and orientation
    may be set via SetOutputSpacing, SetOutputOrigin and
    SetOutputDirection. The default are respectively a vector of 1's, a
    vector of 0's and an identity matrix.

    This class is templated over the type of the input image, the type of
    the output image and the type of the displacement field.

    The input image is set via SetInput. The input displacement field is
    set via SetDisplacementField.

    This filter is implemented as a multithreaded filter.


    WARNING:
    This filter assumes that the input type, output type and displacement
    field type all have the same number of dimensions.

    See:
     itk::simple::Warp for the procedural interface

     itk::WarpImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkWarpImageFilter.h

    
### sitk.WhiteTopHat
    WhiteTopHat(Image arg1, uint32_t radius=1, itk::simple::KernelEnum kernel, bool safeBorder=True) -> Image
    WhiteTopHat(Image arg1, VectorUInt32 vectorRadius, itk::simple::KernelEnum kernel, bool safeBorder=True) -> Image



    itk::simple::WhiteTopHatImageFilter Functional Interface

    This function directly calls the execute method of WhiteTopHatImageFilter in order to support a fully functional API


    
### sitk.WhiteTopHatImageFilter


    White top hat extracts local maxima that are larger than the
    structuring element.


    Top-hats are described in Chapter 4.5 of Pierre Soille's book
    "Morphological Image Analysis: Principles and Applications", Second
    Edition, Springer, 2003.


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.

    See:
     itk::simple::WhiteTopHat for the procedural interface

     itk::WhiteTopHatImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkWhiteTopHatImageFilter.h

    
### sitk.WienerDeconvolution
    WienerDeconvolution(Image image1, Image image2, double noiseVariance=0.0, bool normalize=False, itk::simple::WienerDeconvolutionImageFilter::BoundaryConditionType boundaryCondition, itk::simple::WienerDeconvolutionImageFilter::OutputRegionModeType outputRegionMode) -> Image



    The Wiener deconvolution image filter is designed to restore an image
    convolved with a blurring kernel while keeping noise enhancement to a
    minimum.


    This function directly calls the execute method of WienerDeconvolutionImageFilter in order to support a procedural API


    See:
     itk::simple::WienerDeconvolutionImageFilter for the object oriented interface



    
### sitk.WienerDeconvolutionImageFilter


    The Wiener deconvolution image filter is designed to restore an image
    convolved with a blurring kernel while keeping noise enhancement to a
    minimum.


    The Wiener filter aims to minimize noise enhancement induced by
    frequencies with low signal-to-noise ratio. The Wiener filter kernel
    is defined in the frequency domain as $W(\omega) = H^*(\omega) / (|H(\omega)|^2 + (1 /
    SNR(\omega)))$ where $H(\omega)$ is the Fourier transform of the blurring kernel with which the
    original image was convolved and the signal-to-noise ratio $SNR(\omega)$ . $SNR(\omega)$ is defined by $P_f(\omega) / P_n(\omega)$ where $P_f(\omega)$ is the power spectral density of the uncorrupted signal and $P_n(\omega)$ is the power spectral density of the noise. When applied to the input
    blurred image, this filter produces an estimate $\hat{f}(x)$ of the true underlying signal $f(x)$ that minimizes the expected error between $\hat{f}(x)$ and $f(x)$ .

    This filter requires two inputs, the image to be deconvolved and the
    blurring kernel. These two inputs can be set using the methods
    SetInput() and SetKernelImage() , respectively.

    The power spectral densities of the signal and noise are typically
    unavailable for a given problem. In particular, $P_f(\omega)$ cannot be computed from $f(x)$ because this unknown signal is precisely the signal that this filter
    aims to recover. Nevertheless, it is common for the noise to have a
    power spectral density that is flat or decreasing significantly more
    slowly than the power spectral density of a typical image as the
    frequency $\omega$ increases. Hence, $P_n(\omega)$ can typically be approximated with a constant, and this filter makes
    this assumption (see the NoiseVariance member variable). $P_f(\omega)$ , on the other hand, will vary with input. This filter computes the
    power spectral density of the input blurred image, subtracts the power
    spectral density of the noise, and uses the result as the estimate of $P_f(\omega)$ .

    For further information on the Wiener deconvolution filter, please see
    "Digital Signal Processing" by Kenneth R. Castleman, Prentice Hall,
    1995


    Gaetan Lehmann, Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France

    Chris Mullins, The University of North Carolina at Chapel Hill

    Cory Quammen, The University of North Carolina at Chapel Hill

    See:
     itk::simple::WienerDeconvolution for the procedural interface

     itk::WienerDeconvolutionImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkWienerDeconvolutionImageFilter.h

    
### sitk.WrapPad
    WrapPad(Image image1, VectorUInt32 padLowerBound, VectorUInt32 padUpperBound) -> Image



    Increase the image size by padding with replicants of the input image
    value.


    This function directly calls the execute method of WrapPadImageFilter in order to support a procedural API


    See:
     itk::simple::WrapPadImageFilter for the object oriented interface



    
### sitk.WrapPadImageFilter


    Increase the image size by padding with replicants of the input image
    value.


    WrapPadImageFilter changes the image bounds of an image. Added pixels are filled in with
    a wrapped replica of the input image. For instance, if the output
    image needs a pixel that is two pixels to the left of the
    LargestPossibleRegion of the input image, the value assigned will be
    from the pixel two pixels inside the right boundary of the
    LargestPossibleRegion. The image bounds of the output must be
    specified.

    Visual explanation of padding regions. This filter is implemented as a
    multithreaded filter. It provides a ThreadedGenerateData() method for
    its implementation.


    See:
     MirrorPadImageFilter , ConstantPadImageFilter

     itk::simple::WrapPad for the procedural interface

     itk::WrapPadImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkWrapPadImageFilter.h

    
### sitk.WriteImage
    WriteImage(Image image, std::string const & fileName, bool useCompression=False)
    WriteImage(Image image, VectorString fileNames, bool useCompression=False)



    
### sitk.WriteTransform
    WriteTransform(Transform transform, std::string const & filename)



    
### sitk.Xor
    Xor(Image image1, Image image2) -> Image
    Xor(Image image1, int constant) -> Image
    Xor(int constant, Image image2) -> Image



    
### sitk.XorImageFilter


    Computes the XOR bitwise operator pixel-wise between two images.


    This class is templated over the types of the two input images and the
    type of the output image. Numeric conversions (castings) are done by
    the C++ defaults.

    Since the bitwise XOR operation is only defined in C++ for integer
    types, the images passed to this filter must comply with the
    requirement of using integer pixel type.

    The total operation over one pixel will be


    Where "^" is the boolean XOR operator in C++.
    See:
     itk::simple::Xor for the procedural interface

     itk::XorImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkXorImageFilter.h

    
### sitk.YenThreshold
    YenThreshold(Image image, Image maskImage, uint8_t insideValue=1, uint8_t outsideValue=0, uint32_t numberOfHistogramBins=256, bool maskOutput=True, uint8_t maskValue=255) -> Image
    YenThreshold(Image image, uint8_t insideValue=1, uint8_t outsideValue=0, uint32_t numberOfHistogramBins=256, bool maskOutput=True, uint8_t maskValue=255) -> Image



    
### sitk.YenThresholdImageFilter


    Threshold an image using the Yen Threshold.


    This filter creates a binary thresholded image that separates an image
    into foreground and background components. The filter computes the
    threshold using the YenThresholdCalculator and applies that threshold to the input image using the BinaryThresholdImageFilter .


    Richard Beare

    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.
     This implementation was taken from the Insight Journal paper: https://hdl.handle.net/10380/3279 or http://www.insight-journal.org/browse/publication/811


    See:
     HistogramThresholdImageFilter

     itk::simple::YenThreshold for the procedural interface

     itk::YenThresholdImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkYenThresholdImageFilter.h

    
### sitk.ZeroCrossing
    ZeroCrossing(Image image1, uint8_t foregroundValue=1, uint8_t backgroundValue=0) -> Image



    This filter finds the closest pixel to the zero-crossings (sign
    changes) in a signed itk::Image .


    This function directly calls the execute method of ZeroCrossingImageFilter in order to support a procedural API


    See:
     itk::simple::ZeroCrossingImageFilter for the object oriented interface



    
### sitk.ZeroCrossingBasedEdgeDetection
    ZeroCrossingBasedEdgeDetection(Image image1, double variance=1, uint8_t foregroundValue=1, uint8_t backgroundValue=0, double maximumError=0.1) -> Image



    This filter implements a zero-crossing based edge detecor.


    This function directly calls the execute method of ZeroCrossingBasedEdgeDetectionImageFilter in order to support a procedural API


    See:
     itk::simple::ZeroCrossingBasedEdgeDetectionImageFilter for the object oriented interface



    
### sitk.ZeroCrossingBasedEdgeDetectionImageFilter


    This filter implements a zero-crossing based edge detecor.


    The zero-crossing based edge detector looks for pixels in the
    Laplacian of an image where the value of the Laplacian passes through
    zero points where the Laplacian changes sign. Such points often occur
    at "edges" in images i.e. points where the intensity of the image
    changes rapidly, but they also occur at places that are not as easy to
    associate with edges. It is best to think of the zero crossing
    detector as some sort of feature detector rather than as a specific
    edge detector.


    Zero crossings always lie on closed contours and so the output from
    the zero crossing detector is usually a binary image with single pixel
    thickness lines showing the positions of the zero crossing points.

    In this implementation, the input image is first smoothed with a
    Gaussian filter, then the LaplacianImageFilter is applied to smoothed image. Finally the zero-crossing of the
    Laplacian of the smoothed image is detected. The output is a binary
    image.
    Inputs and Outputs
    The input to the filter should be a scalar, itk::Image of arbitrary dimension. The output image is a binary, labeled image.
    See itkZeroCrossingImageFilter for more information on requirements of
    the data type of the output.

    To use this filter, first set the parameters (variance and maximum
    error) needed by the embedded DiscreteGaussianImageFilter , i.e. See DiscreteGaussianImageFilter for information about these parameters. Optionally, you may also set
    foreground and background values for the zero-crossing filter. The
    default label values are Zero for the background and One for the
    foreground, as defined in NumericTraits for the data type of the output image.

    See:
     DiscreteGaussianImageFilter

     LaplacianImageFilter

     ZeroCrossingImageFilter

     itk::simple::ZeroCrossingBasedEdgeDetection for the procedural interface

     itk::ZeroCrossingBasedEdgeDetectionImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkZeroCrossingBasedEdgeDetectionImageFilter.h

    
### sitk.ZeroCrossingImageFilter


    This filter finds the closest pixel to the zero-crossings (sign
    changes) in a signed itk::Image .


    Pixels closest to zero-crossings are labeled with a foreground value.
    All other pixels are marked with a background value. The algorithm
    works by detecting differences in sign among neighbors using city-
    block style connectivity (4-neighbors in 2d, 6-neighbors in 3d, etc.).

    Inputs and Outputs
    The input to this filter is an itk::Image of arbitrary dimension. The algorithm assumes a signed data type
    (zero-crossings are not defined for unsigned data types), and requires
    that operator>, operator<, operator==, and operator!= are defined.

    The output of the filter is a binary, labeled image of user-specified
    type. By default, zero-crossing pixels are labeled with a default
    "foreground" value of itk::NumericTraits<OutputDataType>::OneValue() , where OutputDataType is the data type of the output image. All
    other pixels are labeled with a default "background" value of itk::NumericTraits<OutputDataType>::ZeroValue() .
    Parameters
    There are two parameters for this filter. ForegroundValue is the value
    that marks zero-crossing pixels. The BackgroundValue is the value
    given to all other pixels.

    See:
     Image

     Neighborhood

     NeighborhoodOperator

     NeighborhoodIterator

     itk::simple::ZeroCrossing for the procedural interface

     itk::ZeroCrossingImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkZeroCrossingImageFilter.h

    
### sitk.ZeroFluxNeumannPad
    ZeroFluxNeumannPad(Image image1, VectorUInt32 padLowerBound, VectorUInt32 padUpperBound) -> Image



    Increase the image size by padding according to the zero-flux Neumann
    boundary condition.


    This function directly calls the execute method of ZeroFluxNeumannPadImageFilter in order to support a procedural API


    See:
     itk::simple::ZeroFluxNeumannPadImageFilter for the object oriented interface



    
### sitk.ZeroFluxNeumannPadImageFilter


    Increase the image size by padding according to the zero-flux Neumann
    boundary condition.


    A filter which extends the image size and fill the missing pixels
    according to a Neumann boundary condition where first, upwind
    derivatives on the boundary are zero. This is a useful condition in
    solving some classes of differential equations.

    For example, invoking this filter on an image with a corner like: returns the following padded image:


    Gaetan Lehmann. Biologie du Developpement et de la Reproduction, INRA
    de Jouy-en-Josas, France.

    See:
     WrapPadImageFilter , MirrorPadImageFilter , ConstantPadImageFilter , ZeroFluxNeumannBoundaryCondition

     itk::simple::ZeroFluxNeumannPad for the procedural interface

     itk::ZeroFluxNeumannPadImageFilter for the Doxygen on the original ITK class.


    C++ includes: sitkZeroFluxNeumannPadImageFilter.h

    
### sitk.cvar
### sitk.numpy
NumPy
=====

Provides
  1. An array object of arbitrary homogeneous items
  2. Fast mathematical operations over arrays
  3. Linear Algebra, Fourier Transforms, Random Number Generation

How to use the documentation
----------------------------
Documentation is available in two forms: docstrings provided
with the code, and a loose standing reference guide, available from
`the NumPy homepage <http://www.scipy.org>`_.

We recommend exploring the docstrings using
`IPython <http://ipython.scipy.org>`_, an advanced Python shell with
TAB-completion and introspection capabilities.  See below for further
instructions.

The docstring examples assume that `numpy` has been imported as `np`::

  >>> import numpy as np

Code snippets are indicated by three greater-than signs::

  >>> x = 42
  >>> x = x + 1

Use the built-in ``help`` function to view a function's docstring::

  >>> help(np.sort)
  ... # doctest: +SKIP

For some objects, ``np.info(obj)`` may provide additional help.  This is
particularly true if you see the line "Help on ufunc object:" at the top
of the help() page.  Ufuncs are implemented in C, not Python, for speed.
The native Python help() does not know how to view their help, but our
np.info() function does.

To search for documents containing a keyword, do::

  >>> np.lookfor('keyword')
  ... # doctest: +SKIP

General-purpose documents like a glossary and help on the basic concepts
of numpy are available under the ``doc`` sub-module::

  >>> from numpy import doc
  >>> help(doc)
  ... # doctest: +SKIP

Available subpackages
---------------------
doc
    Topical documentation on broadcasting, indexing, etc.
lib
    Basic functions used by several sub-packages.
random
    Core Random Tools
linalg
    Core Linear Algebra Tools
fft
    Core FFT routines
polynomial
    Polynomial tools
testing
    NumPy testing tools
f2py
    Fortran to Python Interface Generator.
distutils
    Enhancements to distutils with support for
    Fortran compilers support and more.

Utilities
---------
test
    Run numpy unittests
show_config
    Show numpy build configuration
dual
    Overwrite certain functions with high-performance Scipy tools
matlib
    Make everything matrices.
__version__
    NumPy version string

Viewing documentation using IPython
-----------------------------------
Start IPython with the NumPy profile (``ipython -p numpy``), which will
import `numpy` under the alias `np`.  Then, use the ``cpaste`` command to
paste examples into the shell.  To see which functions are available in
`numpy`, type ``np.<TAB>`` (where ``<TAB>`` refers to the TAB key), or use
``np.*cos*?<ENTER>`` (where ``<ENTER>`` refers to the ENTER key) to narrow
down the list.  To view the docstring for a function, use
``np.cos?<ENTER>`` (to view the docstring) and ``np.cos??<ENTER>`` (to view
the source code).

Copies vs. in-place operation
-----------------------------
Most of the functions in `numpy` return a copy of the array argument
(e.g., `np.sort`).  In-place versions of these functions are often
available as array methods, i.e. ``x = np.array([1,2,3]); x.sort()``.
Exceptions to this rule are documented.


### sitk.operatorOperator interface.

This module exports a set of functions implemented in C corresponding
to the intrinsic operators of Python.  For example, operator.add(x, y)
is equivalent to the expression x+y.  The function names are those
used for special methods; variants without leading and trailing
'__' are also provided for convenience.
### sitk.sitkAbortEventint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkAffineint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkAnnulusint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkAnyEventint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkBSplineint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkBSplineResamplerint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkBSplineResamplerOrder1int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkBSplineResamplerOrder2int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkBSplineResamplerOrder3int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkBSplineResamplerOrder4int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkBSplineResamplerOrder5int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkBSplineTransformint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkBallint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkBlackmanWindowedSincint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkBoxint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkComplexFloat32int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkComplexFloat64int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkCompositeint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkCosineWindowedSincint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkCrossint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkDeleteEventint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkDisplacementFieldint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkEndEventint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkEulerint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkFloat32int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkFloat64int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkGaussianint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkHammingWindowedSincint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkIdentityint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkInt16int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkInt32int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkInt64int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkInt8int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkIterationEventint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkLabelGaussianint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkLabelUInt16int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkLabelUInt32int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkLabelUInt64int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkLabelUInt8int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkLanczosWindowedSincint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkLinearint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkMultiResolutionIterationEventint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkNearestNeighborint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkPolygon3int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkPolygon4int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkPolygon5int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkPolygon6int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkPolygon7int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkPolygon8int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkPolygon9int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkProgressEventint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkQuaternionRigidint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkScaleint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkScaleLogarithmicint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkScaleSkewVersorint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkSimilarityint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkStartEventint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkTranslationint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkUInt16int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkUInt32int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkUInt64int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkUInt8int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkUnknownint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkUserEventint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkVectorFloat32int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkVectorFloat64int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkVectorInt16int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkVectorInt32int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkVectorInt64int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkVectorInt8int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkVectorUInt16int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkVectorUInt32int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkVectorUInt64int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkVectorUInt8int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkVersorint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkVersorRigidint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkWallClockint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sitkWelchWindowedSincint(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### sitk.sysThis module provides access to some objects used or maintained by the
interpreter and to functions that interact strongly with the interpreter.

Dynamic objects:

argv -- command line arguments; argv[0] is the script pathname if known
path -- module search path; path[0] is the script directory, else ''
modules -- dictionary of loaded modules

displayhook -- called to show results in an interactive session
excepthook -- called to handle any uncaught exception other than SystemExit
  To customize printing in an interactive session or to install a custom
  top-level exception handler, assign other functions to replace these.

stdin -- standard input file object; used by input()
stdout -- standard output file object; used by print()
stderr -- standard error object; used for error messages
  By assigning other file objects (or objects that behave like files)
  to these, it is possible to redirect all of the interpreter's I/O.

last_type -- type of last uncaught exception
last_value -- value of last uncaught exception
last_traceback -- traceback of last uncaught exception
  These three are only available in an interactive session after a
  traceback has been printed.

Static objects:

builtin_module_names -- tuple of module names built into this interpreter
copyright -- copyright notice pertaining to this interpreter
exec_prefix -- prefix used to find the machine-specific Python library
executable -- absolute path of the executable binary of the Python interpreter
float_info -- a struct sequence with information about the float implementation.
float_repr_style -- string indicating the style of repr() output for floats
hash_info -- a struct sequence with information about the hash algorithm.
hexversion -- version information encoded as a single integer
implementation -- Python implementation information.
int_info -- a struct sequence with information about the int implementation.
maxsize -- the largest supported length of containers.
maxunicode -- the value of the largest Unicode code point
platform -- platform identifier
prefix -- prefix used to find the Python library
thread_info -- a struct sequence with information about the thread implementation.
version -- the version of this interpreter as a string
version_info -- version information as a named tuple
dllhandle -- [Windows only] integer handle of the Python DLL
winver -- [Windows only] version number of the Python DLL
__stdin__ -- the original stdin; don't touch!
__stdout__ -- the original stdout; don't touch!
__stderr__ -- the original stderr; don't touch!
__displayhook__ -- the original displayhook; don't touch!
__excepthook__ -- the original excepthook; don't touch!

Functions:

displayhook() -- print an object to the screen, and save it in builtins._
excepthook() -- print an exception and its traceback to sys.stderr
exc_info() -- return thread-safe information about the current exception
exit() -- exit the interpreter by raising SystemExit
getdlopenflags() -- returns flags to be used for dlopen() calls
getprofile() -- get the global profiling function
getrefcount() -- return the reference count for an object (plus one :-)
getrecursionlimit() -- return the max recursion depth for the interpreter
getsizeof() -- return the size of an object in bytes
gettrace() -- get the global debug tracing function
setcheckinterval() -- control how often the interpreter checks for events
setdlopenflags() -- set the flags to be used for dlopen() calls
setprofile() -- set the global profiling function
setrecursionlimit() -- set the max recursion depth for the interpreter
settrace() -- set the global debug tracing function

### sitk.weakrefWeak reference support for Python.

This module is an implementation of PEP 205:

http://www.python.org/dev/peps/pep-0205/

### sitk.weakref_proxyproxy(object[, callback]) -- create a proxy object that weakly
references 'object'.  'callback', if given, is called with a
reference to the proxy when 'object' is about to be finalized.
