/*
 * DoorDetection.cpp
 * Door detection by Calonder Descriptor and Harris Corners
 * Vision server that provide corba interface
 *  Created on:    May   10, 2010
 *  Last Modified: Sept. 29, 2010
 *      Author: altrobot
 * Note: 
 *  1. You should obtain the permission to read video1394/raw1394 (sudo chmod 777 /dev/video1394/0).
 *  2. On the robot, capturing two cameras at the same time does not work due to a bandwidth problem.
 *       So after capturing the first camera, release it then capture the second.
 *  3. Image conversion. The image of robot should be firstly converted into Frame (3 channel) to Bayer (1 channel). Don't know why.
 *  4. Now pgm format is used instead of png.
 */

// opencv graphics
#include <highgui.h>
// for stereo vision
#include <frame_common/frame.h>
#include <frame_common/sparse_stereo.h>
// for feature extraction
#include <features_2d/features_2d.h>
#include <boost/shared_ptr.hpp>
#include <cstdio>
// for file read and write
#include <fstream>
#include <dirent.h>
#include <fnmatch.h>
#include <iostream>
// for image capture in the separate thread
#include <pthread.h>
// for processing IEEE cameras
#include <dc1394/dc1394.h>
// Corba idl header
#include "../corba/HRP2Vision.hh"
// elapsed time in milliseconds
#include <sys/time.h>

// declare namespace for convenience
namespace f2d = features_2d;
using namespace frame_common;
using namespace std;
using namespace HRP2Vision;

// coefficients for stereo vision (narrow)
const double V_VIEW_ANGLE= 0.288; // 33 degrees
const int IMAGE_WIDTH= 640;
const int IMAGE_HEIGHT= 480;
const int FRAME_RATE= 30;    // 30 frames per second
// max number of file length including a directory path to a file
const int MAX_FILE_PATH= 200;
// the number of element for the pose of door and knob (x,y,z)
const int NUM_ELEM_TARGET_POSE= 3;
// the number of elements for the transfomration matrix (excluding the last row; 0 0 0 1)
const int NUM_ELEM_TRANSFORM_MAT= 12;
// the upper and lower thresholds for the height of knob from the floor
const double MIN_HEIGHT= 0.8;   // 0.8 meter
const double MAX_HEIGHT= 1.2;   // 1.2 meter

// global variables for cameras, left is 0, right is 1.
dc1394camera_t *gCamera0=NULL, *gCamera1=NULL;
dc1394video_frame_t* gFrame0=NULL, *gFrame1=NULL;
dc1394_t* dc_handle;
dc1394camera_list_t* list;
dc1394error_t err;
// output format of captured image; pgm is much faster to save image than png
string cam0filefmt("Output/left_%04d_%c.pgm");
char cam0filename[MAX_FILE_PATH];
string cam1filefmt("Output/right_%04d_%c.pgm");
char cam1filename[MAX_FILE_PATH];
// variables for keeping the raw image (gray) after conversion from Bayer to Gray
cv::Mat gRawImg0, gRawImg1;
int gCaptureCount= 0;
bool gDoCapture= 0;
bool gCaptureForMap= true;
int gCaptureForMapTimeCount= 0;
// thread to capture image periodically
pthread_t gCaptureThread;

// structure data for the argument of the capture thread
struct THREAD_ARGS
{
	bool RecordFlag;
	bool VisualizeFlag;
};

/*
 * Measure current time in miliseconds.
 * @return the current time in ms.
 */
static double mstime()
{
  timeval tv;
  gettimeofday(&tv,NULL);
  long long ts = tv.tv_sec;
  ts *= 1000000;
  ts += tv.tv_usec;
  return (double)ts*.001;
}

// Names of left and right files in directory (with wildcards)
char *lreg, *rreg;

/************* not used ************************
// Filters for scandir
int getleft(struct dirent const *entry)
{
  if (!fnmatch(lreg,entry->d_name,0))
    return 1;
  return 0;
}

int getright(struct dirent const *entry)
{
  if (!fnmatch(rreg,entry->d_name,0))
    return 1;
  return 0;
}
*/

/*
 * Get the mean of an one dimensional vector (double type)
 * @param aData one dimensional vector of std vector type
 * @return mean in double type
 */
double getMean( const std::vector<double>& aData )
{
	double mean;
	double sum= 0;
	for( std::vector<double>::const_iterator it=aData.begin(); it!=aData.end(); it++ )
		sum+= *it;	
	mean= sum/aData.size();
	return mean;
}

/* 
 * Get the variance of an one dimensional vector (double type)
 * @param aData one dimensional vector of std vector type
 * @param aMean mean of the data
 * @return variance (double type)
*/ 
double getVariance( const std::vector<double>& aData, const double aMean )
{
	double variance;
	double var_sum= 0;
	for( std::vector<double>::const_iterator it=aData.begin(); it!=aData.end(); it++ )
		var_sum+= ( *it-aMean )*( *it-aMean );

	variance= var_sum/aData.size();
	return variance;
}

////////////////////// DC1394 Functions (IEEE Cameras) ////////////////////////////////////
const int NUM_CAMERAS= 2;     // narrow stereo vision 
const int MAX_NUM_CAMERAS= 4; // 4 cameras on the robot

/* narrow stereo vision paramter */
const uint64_t GUID_NV_LEFT_CAM=   0x00b09d01003f7ad4LLU; //49712223525763796LLU;   // LLU is to prevent compile error. Long Long Unsigned
const uint64_t GUID_NV_RIGHT_CAM=  0x00b09d01003f7abbLLU; //49712223525763771LLU;

/*
 * Stop the camera and free the memory.
 * @param camera pointer to the camera.
*/
static void cleanup_and_exit(dc1394camera_t *camera)
{
    dc1394_video_set_transmission(camera, DC1394_OFF);
    dc1394_capture_stop(camera);
    dc1394_camera_free(camera);
    exit(1);
}

/*
 * Set up the camera to capture.
 * @param camera Instance of a camera
 * @return error
 */ 
static dc1394error_t setup_capture( dc1394camera_t * camera )
{
  dc1394error_t err;

  err = dc1394_camera_reset(camera);
  DC1394_ERR_RTN(err, "Cannot initialize camera to factory default settings");


  // iso speed
  err= dc1394_video_set_iso_speed(camera, DC1394_ISO_SPEED_400 );
  DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Could not set iso speed");


  // set video mode
  err= dc1394_video_set_mode(camera, DC1394_VIDEO_MODE_640x480_MONO8 );
  DC1394_ERR_CLN_RTN( err,cleanup_and_exit(camera), "Could not set video mode\n" );

  // set frame rate
  err= dc1394_video_set_framerate(camera, DC1394_FRAMERATE_30 );
  DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Could not set the framerate\n" ); 

  // setup capture	
  const int NUM_DMA_BUFFERS= 16;
  err= dc1394_capture_setup( camera, NUM_DMA_BUFFERS, DC1394_CAPTURE_FLAGS_DEFAULT );
  if(err != DC1394_SUCCESS)
  {
    // after the shutdown is not done correctly, dc1394_capture_setup might fail, so reset the bus.
    DC1394_WRN( err, "Maybe an unclean shutdown was occured in the previous camera capturing.\nTring to reinitialize the bus...\n");
    dc1394_reset_bus(camera);
    // wait a second
    sleep(1);
	  err= dc1394_capture_setup( camera, NUM_DMA_BUFFERS, DC1394_CAPTURE_FLAGS_DEFAULT );
  }	
  
  if(err != DC1394_SUCCESS)
  {
    return err;
  }

  sleep(1);

  /// flush the ring buffer
  for( int i = 0; i <NUM_DMA_BUFFERS; i++ )
  {
    dc1394video_frame_t* frame = NULL;
    if( dc1394_capture_dequeue(camera, DC1394_CAPTURE_POLICY_POLL, &frame ) == DC1394_SUCCESS )
    {
      if(frame)
      {
        dc1394_capture_enqueue(camera, frame);
      }
    }
  }

  /// Set features
  // set brightness
  dc1394_feature_set_mode( camera, DC1394_FEATURE_BRIGHTNESS, DC1394_FEATURE_MODE_MANUAL );
  err= dc1394_feature_set_value( camera, DC1394_FEATURE_BRIGHTNESS, 200 );
  DC1394_ERR_CLN_RTN( err, cleanup_and_exit(camera), "Could not set brightness" );

   // set exposure
  dc1394_feature_set_mode( camera, DC1394_FEATURE_EXPOSURE, DC1394_FEATURE_MODE_MANUAL );
  err= dc1394_feature_set_value( camera, DC1394_FEATURE_EXPOSURE, 352 );
  DC1394_ERR_CLN_RTN( err, cleanup_and_exit(camera), "Could not set exposure" );

	/************************** NOTE ********************************************/
  /* gain and the shutter speed are very important paramters to change the brightness and the image blur.
  /* increase the gain and decrease shutter speed to get less blurred image with brightness. -> images become noisy
  /* decrease the gain and increase shutter speed to get less noisy image.
  /****************************************************************************/
  // set gain	
  dc1394_feature_set_mode(camera,DC1394_FEATURE_GAIN,DC1394_FEATURE_MODE_MANUAL);
  err=dc1394_feature_set_value(camera, DC1394_FEATURE_GAIN,600);   // for knob recognition
//  err=dc1394_feature_set_value(camera, DC1394_FEATURE_GAIN, 900 ); // for visual mapping
  DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Could not set up gain");

  // set shutter
  dc1394_feature_set_mode(camera,DC1394_FEATURE_SHUTTER,DC1394_FEATURE_MODE_MANUAL);
  err=dc1394_feature_set_value(camera,  DC1394_FEATURE_SHUTTER,650);  // for knob recognition
//  err=dc1394_feature_set_value(camera,  DC1394_FEATURE_SHUTTER, 300);  // for visual mapping
  DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Could not set up shutter");

  // set white balance
  dc1394_feature_set_mode(camera,DC1394_FEATURE_WHITE_BALANCE, DC1394_FEATURE_MODE_MANUAL);
  dc1394_feature_whitebalance_set_value( camera, 90, 80 );
  DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Could not set up white balance");
  
  // set gamma
  dc1394_feature_set_mode(camera,DC1394_FEATURE_GAMMA, DC1394_FEATURE_MODE_MANUAL);
  err=dc1394_feature_set_value(camera, DC1394_FEATURE_GAMMA, 1024);
  DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Could not set up gamma");

  return DC1394_SUCCESS;
}

/*
 * A thread that captures images from cameras periodically.
 * @param aArgs the arguments for the thread:
 *        record flag: to save images as files
 *        visualation flag: to display the result on the screen
 */
void *captureFromCamera( void* aArgs )
{
	// convert the argument in the original form
	struct THREAD_ARGS* args;
	args= (THREAD_ARGS*)aArgs;

	bool recordFlag= args->RecordFlag;
	bool visualizeFlag= args->VisualizeFlag;
  
	// setup sleep time
	struct timespec req={0},rem={0};
	req.tv_sec= 0;
	req.tv_nsec= 33333333; // 30Hz = 33.333ms 

	double cur_time;
	// capture images periodically until the robot stops
 	while( gDoCapture )
	{
		cur_time= mstime();
		// capture an image from the left camera
    err= dc1394_capture_dequeue( gCamera0, DC1394_CAPTURE_POLICY_WAIT, &gFrame0 );
    if (err!=DC1394_SUCCESS)
	  {
    	dc1394_log_error("unable to capture from the left camera");
    	dc1394_capture_stop( gCamera0 );
    	dc1394_camera_free( gCamera0 );
    	break;
    }
		// capture an image from the right camera
		err= dc1394_capture_dequeue( gCamera1, DC1394_CAPTURE_POLICY_WAIT, &gFrame1 );
   	if (err!=DC1394_SUCCESS)
  	{
    	dc1394_log_error("unable to capture from the right camera");
    	dc1394_capture_stop( gCamera1 );
    	dc1394_camera_free( gCamera1 );
    	break;
   	}

		cv::Mat bayerImg0, bayerImg1, colorImg0, colorImg1;		// convert the rgb to gray image to get one-channel image
	
 		// convert image into opencv type image, which is a bayer format
 		bayerImg0= cv::Mat( IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, gFrame0->image );
 		bayerImg1= cv::Mat( IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, gFrame1->image );

		// convert bayer format to rgb format to extract the original image three channel.
		cv::cvtColor( bayerImg0, colorImg0, CV_BayerGR2RGB );   // get the color image from bayer format
		cv::cvtColor( bayerImg1, colorImg1, CV_BayerGR2RGB );   // get the color image from bayer format
 
		// convert the rgb to gray image to get one-channel image
		cv::cvtColor( colorImg0, gRawImg0, CV_RGB2GRAY );   // get the gray image
		cv::cvtColor( colorImg1, gRawImg1, CV_RGB2GRAY );   // get the gray image

		if( recordFlag ) 		// save image as pgm format
		{	
			if( gCaptureForMap )
			{	
				sprintf( cam0filename, cam0filefmt.c_str(), gCaptureCount, 'D' );
				sprintf( cam1filename, cam1filefmt.c_str(), gCaptureCount, 'D' );
				gCaptureForMap= false;

				cout << "time count after becoming Double Support: " << gCaptureForMapTimeCount << endl;
				cout << "the file name: " << cam0filename << endl << endl; 
			}
			else
			{
				sprintf( cam0filename, cam0filefmt.c_str(), gCaptureCount, 'S' );
				sprintf( cam1filename, cam1filefmt.c_str(), gCaptureCount, 'S' );
			}
						
    		FILE* imagefile= fopen( cam0filename, "wb" );
    		fprintf( imagefile, "P5\n%u %u\n255\n", IMAGE_WIDTH, IMAGE_HEIGHT );
    		fwrite( gRawImg0.data, 1, IMAGE_WIDTH*IMAGE_HEIGHT, imagefile );
    		fclose( imagefile );

    		imagefile= fopen( cam1filename, "wb" );
	   		fprintf( imagefile, "P5\n%u %u\n255\n", IMAGE_WIDTH, IMAGE_HEIGHT );
    		fwrite( gRawImg1.data, 1, IMAGE_WIDTH*IMAGE_HEIGHT, imagefile );
    		fclose( imagefile );
		}    	

		// release buffer
 		dc1394_capture_enqueue( gCamera0, gFrame0 );
   	dc1394_capture_enqueue( gCamera1, gFrame1 );
		
		// increaes the capture count
		gCaptureCount++;

		// sleep for the defined time
//	  	nanosleep( &req, &rem );     // this is not necessary since the frame rate is set to 30Hz.
	
  }
	
  pthread_exit(0);
}
////////////////////// END of Capture Functions //////////////////////////////

// base images for door recognition
const char* BASE_L_DOOR= "etc/base_door.pgm";
const char* BASE_R_DOOR= "etc/door_right.pgm";
// knob images are used only for the dummy function -> It should be modified since these are not necessary.
const char* BASE_L_KNOB= "etc/base_knob.pgm";
const char* BASE_R_KNOB= "etc/knob_right.pgm";
// learned rtc (randomized tree classifier) by ROS (I don't know which images are used for the learning)
// this works too but not as good as the door.rtc.
//const char* CALONDER_TREE_FILE= "etc/calonder.rtc";
// learned rtc by Nosan using an image (5000 different views, 80 trees)
const char* CALONDER_TREE_FILE= "etc/door.rtc";

////////////////////////////// Services of Vision Computer /////////////////////////////////////////////
class HRP2Service :  public POA_HRP2Vision::Door,
                     public PortableServer::RefCountServantBase
{
public:

  // CORBA Serivce Implementation	
	virtual CORBA::Boolean setBaseImage( CORBA::Short aImgID );
  virtual CORBA::Boolean getTargetPose( CORBA::Float* theTargetPose );
  virtual CORBA::Boolean getKnobPose( CORBA::Double aDist2Door, CORBA::Double* theKnobPose );
  virtual void           setVisualization( CORBA::Boolean aVisualize );
  virtual void           setRecord( CORBA::Boolean aVideoRecord );
  virtual void setCaptureForMap( CORBA::Boolean aCapture );
  virtual void setCaptureForMapWithTime( CORBA::Boolean aCapture, CORBA::Short aTimeCount );
 	virtual	CORBA::Boolean setBaseCoordinate( const CORBA::Float* theTransformMat );

  // Member Functions
  HRP2Service( int aBaseImgID=0 );
  virtual ~HRP2Service();
  bool initNarrowVision();
  bool matchScene();
	bool computeTargetPose();
	Eigen::Vector4d extractKnob( double aDist2Door, cv::Mat& aImg, const std::string win_name );
	bool checkHeightContraint( Eigen::Vector2d aCentroid, double aCameraHeight, double aDist2Door );
	bool computeKnobPose();

private:
	int mCallCount;
	fstream mLogFile;
  float* mTargetPose;
  bool mVisualizeFlag;
  bool mRecordFlag;

	int mLeftNCamID, mRightNCamID;

  cv::Size mImgSize;
	cv::Mat mBaseImg0;
	cv::Mat mBaseImg1;
  CamParams mCamParam;
	cv::Mat mRectImg0, mRectImg1;

	// HRP2 Kinematics
	float* mBaseTransformMat;    // transform matrix from the left ankle (LLEG_JOINT3) to the head (HEAD_JOINT1)

	// for rectification
  cv::Mat M1, D1, M2, D2, R, T, R1, P1, R2, P2;

	// for feature point and descriptor  
	boost::shared_ptr<f2d::FeatureDetector> mDetector;
  boost::shared_ptr<f2d::DescriptorExtractor> mExtractor;
  std::vector<cv::KeyPoint> mBaseKeypoints; // key points for base image 
	std::vector<cv::KeyPoint> mMatchPoints;   // matched points 
	cv::Mat mBaseDescriptors;   

  // ratio of true_length/num_pixels
	double m_kappa;
};

// file to save the long that is printed on the screen
const char* LOG_FILE= "Output/log.txt";

/*
 * Constructor of HRP2Service.
 * @param aBaseImgID id of the base image; 0 for door and 1 for the knob
 */
HRP2Service::HRP2Service( int aBaseImgID )
{
	mLeftNCamID= mRightNCamID= -1;

	mCallCount= 0;
	m_kappa= 0.0;
	mLogFile.open( LOG_FILE, fstream::out );
	mTargetPose= new float[NUM_ELEM_TARGET_POSE];	
	mBaseTransformMat= new float[NUM_ELEM_TRANSFORM_MAT];	
	mImgSize= cv::Size( IMAGE_WIDTH, IMAGE_HEIGHT );
	mVisualizeFlag= false;
	mRecordFlag= false;
	mMatchPoints.clear();

	// initialize the stereo vision (narrow)
	if( !initNarrowVision() )
	{
		cerr << "********** Narrow Vision couldn't be initialized **************" << endl;
		mLogFile << "********** Narrow Vision couldn't be initialized **************" << endl;
	}
	else
	{
		cout << "Narrow Vision is initialized." << endl;
		mLogFile << "Narrow Vision is initialized." << endl;
	}
}
                                          
/*
 * Destructor of HRP2Service.
 */                                      
HRP2Service::~HRP2Service()
{
	if( mTargetPose )
		delete[] mTargetPose;


	// stop transmission
  err= dc1394_video_set_transmission( gCamera0,DC1394_OFF );
//  DC1394_ERR_CLN_RTN(err,cleanup_and_exit(gCamera0),"Could not stop the camera");
  err= dc1394_video_set_transmission( gCamera1,DC1394_OFF );
//  DC1394_ERR_CLN_RTN(err,cleanup_and_exit(gCamera1),"Could not stop the camera"); 

	// close camera
  dc1394_capture_stop( gCamera0 );
  dc1394_camera_free( gCamera1 );
  dc1394_capture_stop( gCamera0 );
  dc1394_camera_free( gCamera1 );

  dc1394_free( dc_handle );

}

/////////////////////////// [BEGIN] CORBA Services /////////////////////////////////////////////
/*
 * Compute the door position using stereo vision.
 * @param theTargetPose the resulting pose of the target (door).
 * @return false when the scene is not matched.
 */
CORBA::Boolean HRP2Service::getTargetPose( CORBA::Float* theTargetPose )
{
	cout << "Target Pose is requested" << endl;	
	mLogFile << "Target Pose is requested" << endl;	
	bool bMatched= matchScene();

	if( !bMatched )
	{
		cout << "Images are not matched" << endl;
		mLogFile << "Images are not matched" << endl;
		return false;
	}
	if( computeTargetPose() )
	{
		for( int i=0; i<NUM_ELEM_TARGET_POSE; i++ )
			theTargetPose[i]= mTargetPose[i];
		return true;
	}
	else
		return false;
}


/*
 * Compute the pose of knob.
 * @param theKnobPose resulting pose of the knob.
 * @return constraint of the height of knob is check.
 */
CORBA::Boolean HRP2Service::getKnobPose( CORBA::Double aDist2Door, CORBA::Double* theKnobPose )
{
	cout << "Knob Pose is requested" << endl;	
	mLogFile << "Knob Pose is requested" << endl;	


	// Matching for the knob scene is not necessary. [TODO] remove this part but be careful of the matchScene function.
  // For the moment, it is called to get rectified images that are obtained in the matchScene function.
	bool bMatched= matchScene();
	if( !bMatched )
	{
		cerr << "Images of knob are not matched" << endl;
		mLogFile << "Images of knob are not matched" << endl;
//		return false;
	}
	
  // extract the knob from the left image (the image should be rectified)
	double t1= mstime();
	Eigen::Vector4d centroid0= extractKnob( aDist2Door, mRectImg0, "left_knob" );	
	if( centroid0(0) == 0.0 ) // knob is not detected.
	{
		cerr << "Knob could not be extracted from the left image." << endl;
		mLogFile << "Knob could not be extracted from the left image." << endl;
		return false;
	}

	// measure extraction time
	cout << "Left knob is detected in " << mstime()-t1 << " [ms]." << endl;
	mLogFile << "Left knob is detected in " << mstime()-t1 << " [ms]." << endl;

	cout << "centroid (c,r,w,h) from the left " << centroid0(0) << ", " <<  centroid0(1) << ", " << centroid0(2) << ", " << centroid0(3) << endl;
	mLogFile << "centroid (c,r,w,h) from the left " << centroid0(0) << ", " <<  centroid0(1) << ", " << centroid0(2) << ", " << centroid0(3) << endl;
	double x_l_c= M1.at<double>(0,2);     // the principal x center of the left iamge
	double x_l_d= centroid0(0) - x_l_c;      
  // extract the knob from the right image (the image should be rectified)
	double t2= mstime();	
	Eigen::Vector4d centroid1= extractKnob( aDist2Door, mRectImg1, "right_knob" );
	if( centroid1(0) == 0.0 )
	{
		cerr << "Knob could not be extracted from the right image." << endl;
		mLogFile << "Knob could not be extracted from the right image." << endl;
		return false;
	}

	// measure extraction time
	cout << "Right knob is detected in " << mstime()-t2 << " [ms]." << endl;
	mLogFile << "Right knob is detected in " << mstime()-t2 << " [ms]." << endl;


	cout << "centroid (x,y,z,w,h) from the right " << centroid1(0) << ", " <<  centroid1(1) << ", " << centroid1(2) << ", " << centroid1(3) << endl;
	mLogFile << "centroid (x,y,z,w,h) from the right " << centroid1(0) << ", " <<  centroid1(1) << ", " << centroid1(2) << ", " << centroid1(3) << endl;
	double x_r_c= M2.at<double>(0,2);     // the pricipal x center of the right image
	double x_r_d= centroid1(0) - x_r_c;

	double disparity= x_l_d - x_r_d;
	cout << "disparity of the centroild of the blob is " << disparity << endl;
	mLogFile << "disparity of the centroild of the blob is " << disparity << endl;

	// to stop computing and to display the view
//	char key= cv::waitKey();
//	if( key == 27 ) return false;

	// to measure the 3d position of the knob
	double t3= mstime();

 	// compute the 3d point of the centroid pixel of the blob, based on camera frame (coordinate)
 	Eigen::Vector3d camTarget;
	double x= centroid0(0) - mCamParam.cx;
	double y= centroid0(1) - mCamParam.cy;
	double z= mCamParam.fx;
	double w= disparity/mCamParam.tx;
	camTarget= Eigen::Vector3d( x/w, y/w, z/w );

	cout << "knob (x,y,z)= " << camTarget(0) << ", " << camTarget(1) << ", " << camTarget(2) << endl; 
	mLogFile << "knob (x,y,z)= " << "( " << camTarget(0) << ", " << camTarget(1) << ", " << camTarget(2) << " )" << endl;

	/// Get the pose based on the base coordinate
	// sanity check
	if( mBaseTransformMat[11] < 0 || mBaseTransformMat[11] > 1.5 )
	{
		cerr << "the transform matrix is not correct" << endl;
		mLogFile << "the transform matrix is not correct" << endl;
		return false;
	}

	// print the matrix
	cout << mBaseTransformMat[0] << " " << mBaseTransformMat[1] << " " << mBaseTransformMat[2] << " " << mBaseTransformMat[3] << endl;
	cout << mBaseTransformMat[4] << " " << mBaseTransformMat[5] << " " << mBaseTransformMat[6] << " " << mBaseTransformMat[7] << endl;
	cout << mBaseTransformMat[8] << " " << mBaseTransformMat[9] << " " << mBaseTransformMat[10] << " " << mBaseTransformMat[11] << endl;
	
	mLogFile << mBaseTransformMat[0] << " " << mBaseTransformMat[1] << " " << mBaseTransformMat[2] << " " << mBaseTransformMat[3] << endl;
	mLogFile << mBaseTransformMat[4] << " " << mBaseTransformMat[5] << " " << mBaseTransformMat[6] << " " << mBaseTransformMat[7] << endl;
	mLogFile << mBaseTransformMat[8] << " " << mBaseTransformMat[9] << " " << mBaseTransformMat[10] << " " << mBaseTransformMat[11] << endl;

  // express the knob pose based on the frame that is under the left foot.
	Eigen::Vector3d baseTarget;
	baseTarget(0)= mBaseTransformMat[0]*camTarget(0) + mBaseTransformMat[1]*camTarget(1) + mBaseTransformMat[2]*camTarget(2) + mBaseTransformMat[3];
	baseTarget(1)= mBaseTransformMat[4]*camTarget(0) + mBaseTransformMat[5]*camTarget(1) + mBaseTransformMat[6]*camTarget(2) + mBaseTransformMat[7];
	baseTarget(2)= mBaseTransformMat[8]*camTarget(0) + mBaseTransformMat[9]*camTarget(1) + mBaseTransformMat[10]*camTarget(2) + mBaseTransformMat[11];

	cout << "target based on the ground frame: " << baseTarget << endl;
	mLogFile << "target based on the ground frame: " << baseTarget << endl;

	// check if the height of blob satisfies the height constraint
	double height= baseTarget(2);
	if( height < MIN_HEIGHT || height > MAX_HEIGHT )
	{
		cerr << "height of the knob: " << height << ", The blob is not in the range of standard height" << endl;
  	    mLogFile << "height of the knob: " << height << ", The blob is not in the range of standard height" << endl;
		return false;
	}

	// measure the computation of the 3d position of the knob
	cout << "3d position of the knob is computed in " << mstime()-t3 << " [ms]." << endl;
	mLogFile << "3d position of the knob is computed in " << mstime()-t3 << " [ms]." << endl;

	
	cout << "height of the knob: " << height << endl;
	mLogFile << "height of the knob: " << height << endl;

	// return the position of the knob based on the LLEG (ultimately, the base frame should be waist joint )
	theKnobPose[0]= baseTarget(0);
	theKnobPose[1]= baseTarget(1);
	theKnobPose[2]= baseTarget(2);

	// re-compute the ratio of true_length/num_pixels, m_kappa for the better width and height of the knob. 
  // The computation is done on the camera frame to remove the transformation error
	m_kappa= camTarget(2)*tan(V_VIEW_ANGLE/2)/mCamParam.cy;
 	cout << "[getKnobPose] kappa= " << m_kappa << endl;
 	mLogFile << "[getKnobPose] kappa= " << m_kappa << endl;

	theKnobPose[3]= m_kappa*( centroid0(2) + centroid1(2) )/2.0;   // estimated width of the knob
	theKnobPose[4]= m_kappa*( centroid0(3) + centroid1(3) )/2.0;   // estimated height of the knob

	cout << "knob (x,y,z,w,h): " << theKnobPose[0] << ", " << theKnobPose[1] << ", " << theKnobPose[2] << ", " << theKnobPose[3] << ", " << theKnobPose[4] << endl;
	mLogFile << "knob (x,y,z,w,h): " << theKnobPose[0] << ", " << theKnobPose[1] << ", " << theKnobPose[2] << ", " << theKnobPose[3] << ", " << theKnobPose[4] << endl;

	return true;
}

/* 
 * Set the base image for matching.
 * There are two types of base images: door and knob.
 * @param aImgID 0 for base image of door and 1 for knob.
 * @return true if successful
 */
CORBA::Boolean HRP2Service::setBaseImage( CORBA::Short aImgID )
{
  // Load base image (left image to be used as a base image )
	switch( aImgID )
	{
	   	case 0: mBaseImg0= cv::imread( BASE_L_DOOR, 0 ); 
				 mBaseImg1= cv::imread( BASE_R_DOOR, 0 ); 		
						break;
		case 1: mBaseImg0= cv::imread( BASE_L_KNOB, 0 ); 
				 mBaseImg1= cv::imread( BASE_R_KNOB, 0 );				
						break;
		default: cerr << " the image id is not correct." << endl; 
						return false;
	}
	
	// sanity check
	if( mBaseImg0.data == NULL || mBaseImg1.data == NULL )
	{
		cerr << "Could not open the image" << endl;
		return false;
	}		

	// Rectify base images (Rectified images give the better matching ratio and disparity)
 	cv::Mat Q;	
 	cv::Rect roi1, roi2;
 	cv::Size img_size = mBaseImg0.size(); 
 	cv::stereoRectify( M1, D1, M2, D2, mImgSize, R, T, R1, R2, P1, P2, Q, -1, mImgSize, &roi1, &roi2 );
 	cv::Mat map11, map12, map21, map22;
 	cv::initUndistortRectifyMap( M1, D1, R1, P1, mImgSize, CV_16SC2, map11, map12);
 	cv::initUndistortRectifyMap( M2, D2, R2, P2, mImgSize, CV_16SC2, map21, map22);
 	cv::Mat img1r, img2r;
 	cv::remap( mBaseImg0, img1r, map11, map12, cv::INTER_LINEAR );
 	cv::remap( mBaseImg1, img2r, map21, map22, cv::INTER_LINEAR );
	img1r.copyTo( mBaseImg0 );
	img2r.copyTo( mBaseImg1 );

 	// Calonder tree setup, don't count in computation time
 	f2d::CalonderDescriptorExtractor<float> *cd = new f2d::CalonderDescriptorExtractor<float>( CALONDER_TREE_FILE );

 	// Detect keypoints (Harris Corner)
 	// mDetector= boost::shared_ptr<f2d::FeatureDetector>( new f2d::HarrisFeatureDetector(300, 10) );
 	// Detect keypoints (Star detector)
	mDetector= boost::shared_ptr<f2d::FeatureDetector>( new f2d::StarFeatureDetector );

	// extract features
	mBaseKeypoints.clear();
	mDetector->detect( mBaseImg0, mBaseKeypoints );
	printf("%d keypoints from the base image.\n", mBaseKeypoints.size() );
	mLogFile << mBaseKeypoints.size() << " keypoints from the base image." << endl;

  // Compute descriptors
	mExtractor= boost::shared_ptr<f2d::DescriptorExtractor>(cd);
	mExtractor->compute( mBaseImg0, mBaseKeypoints, mBaseDescriptors );

	return true;	
}

/* 
 * Set visualization flag, which shows resulting images.
 * @param aVisualize 1 for visuzliation and 0 for no visualization.
 */
void HRP2Service::setVisualization( CORBA::Boolean aVisualize )
{
	mVisualizeFlag= aVisualize;
	if( mVisualizeFlag )
	{
		cout << "Turn on Visualization Flag" << endl;
		mLogFile << "Turn on Visualization Flag" << endl;
	} 
	
	else
	{
		cout << "Turn off Visualization Flag" << endl;
		mLogFile << "Turn off Visualization Flag" << endl;
	}
}

/*
 * Set the flag for recording data such as images.
 * @param aRecord 1 for recording and 0 for no recording
 */
void HRP2Service::setRecord( CORBA::Boolean aRecord )
{
	mRecordFlag= aRecord;

	if( mRecordFlag )
	{
		cout << "Turn on Record Flag" << endl;
		mLogFile << "Turn on Record Flag" << endl;

		gDoCapture= 1;
		struct THREAD_ARGS thread_args;
		thread_args.RecordFlag= mRecordFlag;
		thread_args.VisualizeFlag= mVisualizeFlag;
		pthread_create( &gCaptureThread, NULL, captureFromCamera, (void*) &thread_args );
	}
	else
	{
		// stop the capture
		gDoCapture= 0;
		// stop the thread
//  		pthread_join( gCaptureThread, NULL );   // this is a wait function
		cout << "Turn off Record Flag" << endl;
		mLogFile << "Turn off Record Flag" << endl;
	}
}

/*
 * Set the flag for capturing images for building a depth map.
 * Depeding on the flag, the name of saving file changes. 
 * @param aCapture
 */
void HRP2Service::setCaptureForMap( CORBA::Boolean aCapture )
{
	if( aCapture )
	{
		cout << "Turn on CaptureForMap Flag" << endl;
		mLogFile << "Turn on CaptureForMap Flag" << endl;

		gCaptureForMap= true;
	}
	else
	{
//		cout << "Turn off CaptureForMap Flag" << endl;
//		mLogFile << "Turn off CaptureForMap Flag" << endl;
//		gCaptureForMap= false;
	}
}

/*
 * This is a test function to know the best time to capture image for map. 
 * Set the flag for capturing images for building a depth map.
 * Depeding on the flag, the name of saving file changes. 
 * @param aCapture
 */
void HRP2Service::setCaptureForMapWithTime( CORBA::Boolean aCapture, CORBA::Short aTimeCount )
{
	if( aCapture )
	{
		cout << "Turn on CaptureForMap Flag" << endl;
		mLogFile << "Turn on CaptureForMap Flag" << endl;
		gCaptureForMap= true;
		gCaptureForMapTimeCount= aTimeCount;
	}
	else
	{
		cout << "Turn off CaptureForMap Flag" << endl;
		mLogFile << "Turn off CaptureForMap Flag" << endl;
		gCaptureForMap= false;
		gCaptureForMapTimeCount= 0;
	}
}


/*
 * Set the base coordinate of camera frame.
 * @param theTransformMat the transformation matrix from a base (LLEG_JOINT3) to head joint (HEAD_JOINT1)
 * @return the success
 */
CORBA::Boolean HRP2Service::setBaseCoordinate( const CORBA::Float* theTransformMat )
{
	cout << "Setting the base transformation matrix ..." << endl;	
	mLogFile << "Setting the base transformation matrix ..." << endl;	
		
	for( int i=0; i<NUM_ELEM_TRANSFORM_MAT; i++ )
		mBaseTransformMat[i]= theTransformMat[i];

  return true;
}
/////////////////////////// [END] CORBA Services /////////////////////////////////////////////

/*
 * Intialize the narrow vision system of HRP2.V10
 * 1. read camera parameters (intrinsic/extrinsic)
 * 2. setup ieee1394 cameras to capture
 * 3. create a thread to periodically capture images
 * @return
 */
bool HRP2Service::initNarrowVision()
{

	const char* INTRINSICS_FILE= "etc/intrinsics.yml";
	const char* EXTRINSICS_FILE= "etc/extrinsics.yml";

  // reading intrinsic parameters
  cv::FileStorage fs( INTRINSICS_FILE, CV_STORAGE_READ );
  if(!fs.isOpened())
  {
  	  printf("Failed to open file %s\n", INTRINSICS_FILE);
      return false;
  }

  fs["M1"] >> M1;
  fs["D1"] >> D1;
  fs["M2"] >> M2;
  fs["D2"] >> D2;

  // read extrinsic parameters 
  fs.open(EXTRINSICS_FILE, CV_STORAGE_READ);
  if(!fs.isOpened())
  {
       printf("Failed to open file %s\n", EXTRINSICS_FILE);
       return false;
  }
      
  fs["R"] >> R;
  fs["T"] >> T;
  // setup camera parameters
  mCamParam.fx= M1.at<double>(0,0);
  mCamParam.fy= M1.at<double>(1,1);
  mCamParam.cx= M1.at<double>(0,2);
  mCamParam.cy= M1.at<double>(1,2);
  mCamParam.tx= T.at<double>(0,0)*(-1.0); // for camera coordinate (positive into the image direction)

  cout << "Cam params: " << mCamParam.fx << " " << mCamParam.fy << " " << mCamParam.cx
       << " " << mCamParam.cy << " " << mCamParam.tx << endl;

  mLogFile << "Cam params: " << mCamParam.fx << " " << mCamParam.fy << " " << mCamParam.cx
           << " " << mCamParam.cy << " " << mCamParam.tx << endl;

	// clear call count 
	mCallCount= 0;

	// setup for capture
  dc_handle = dc1394_new();

  if( !dc_handle )
    return false;
  err= dc1394_camera_enumerate( dc_handle, &list );
  DC1394_ERR_RTN(err,"Failed to enumerate cameras");

  if (list->num == 0)
  {
    dc1394_log_error( "No cameras found" );
    return false;
  }

  // find camera id using its GUID 
  for( uint32_t i=0; i<list->num; i++ )
	{

   	if( list->ids[i].guid == GUID_NV_LEFT_CAM )
		{
 		  // create camera
		  gCamera0= dc1394_camera_new( dc_handle, list->ids[i].guid );
			mLeftNCamID= i;
 			cout << "Left narrow camera is found, id is: " << i << endl;
			mLogFile << "Left narrow camera is found, id is: " << i << endl;
    }
		else if( list->ids[i].guid == GUID_NV_RIGHT_CAM )
		{
 		  // create camera
		  gCamera1= dc1394_camera_new( dc_handle, list->ids[i].guid );
			mRightNCamID= i;
 			cout << "Right narrow camera is found, id is: " << i << endl;
			mLogFile << "Right narrow camera is found, id is: " << i << endl;
		}
	}

 	// sanity check
	assert( mLeftNCamID != -1 || mRightNCamID != -1 );
	
  // frees the memory allocated in dc1394_enumerate_cameras for the camera list 
 	dc1394_camera_free_list(list);

	// setup capture
	setup_capture( gCamera0 );
	setup_capture( gCamera1 );
  	err= dc1394_video_set_transmission( gCamera0, DC1394_ON );
 	DC1394_ERR_CLN_RTN( err, cleanup_and_exit( gCamera0 ),"Could not start camera iso transmission\n" );
  	err= dc1394_video_set_transmission( gCamera1, DC1394_ON );
  	DC1394_ERR_CLN_RTN( err, cleanup_and_exit( gCamera1 ), "Could not start camera iso transmission\n" );

  return true;
}

/*
 * Match two images: base and input images using calonder descriptor
 */
bool HRP2Service::matchScene()
{
 	double t1 = mstime();

	/// Rectify input images
	cv::Mat Q;	
 	cv::Rect roi1, roi2;
  
  	// rectify images     
	cv::stereoRectify( M1, D1, M2, D2, mImgSize, R, T, R1, R2, P1, P2, Q, -1, mImgSize, &roi1, &roi2 );

 	cv::Mat map11, map12, map21, map22;
 	cv::initUndistortRectifyMap( M1, D1, R1, P1, mImgSize, CV_16SC2, map11, map12 );
 	cv::initUndistortRectifyMap( M2, D2, R2, P2, mImgSize, CV_16SC2, map21, map22 );
        
 	cv::Mat img1r, img2r;
 	cv::remap( gRawImg0, img1r, map11, map12, cv::INTER_LINEAR );
 	cv::remap( gRawImg1, img2r, map21, map22, cv::INTER_LINEAR );
        
	img1r.copyTo( mRectImg0 );
	img2r.copyTo( mRectImg1 );

	// detect keypoints
	std::vector<cv::KeyPoint> keypoints;   // matched points 
  mDetector->detect( mRectImg0, keypoints );
	printf( "%d keypoints from the test image.\n", keypoints.size() );
	mLogFile << keypoints.size() << " keypoints from the test image." << endl;

  // compute descriptors
  cv::Mat descriptors;
  mExtractor->compute( mRectImg0, keypoints, descriptors );
  printf("%d keypoints remaining from the test image.\n", keypoints.size() );
	mLogFile << keypoints.size() << " keypoints remaining from the test image." << endl;

  // perform matching
  boost::shared_ptr<f2d::DescriptorMatcher> matcher(new f2d::BruteForceMatcher< f2d::L2<float> >);
  std::vector<f2d::Match> matches;
	// for larger change between images. Sliding Window size= (160,120)
  matcher->matchWindowed( mBaseKeypoints, mBaseDescriptors, keypoints, descriptors, 160, 120, matches );
	printf("%d matches\n", matches.size());
  mLogFile << matches.size() << " matches" << endl;
  
  // Remove duplicated index in the matches
  std::vector<int> all_indices;
  all_indices.clear();
  vector<int>::iterator it;
  cv::KeyPoint pt;
  for( int i=0; i<(int)matches.size(); i++ )
  {
   	int ind= matches[i].index2;
  	it= std::search( all_indices.begin(), all_indices.end(), &ind, &ind+1 );
		if( it == all_indices.end() )   // not duplicated?
		{
			all_indices.push_back( ind );
			pt= keypoints[ matches[i].index2 ];
			mMatchPoints.push_back( pt );
		}
	}
	printf( "number of real matches: %d\n", all_indices.size() );
  mLogFile << "number of real matches: " << all_indices.size() << " matches" << endl;

  double t2 = mstime();
	cout << "Elapsed time of matchScene(): " << t2-t1 << "[ms]" << endl;
	mLogFile << "Elapsed time of matchScene(): " << t2-t1 << "[ms]" << endl;

	if( mVisualizeFlag ) // visualize matched window
	{
  	cv::Mat display;
  	f2d::drawMatches( mBaseImg0, mBaseKeypoints, mRectImg0, keypoints, matches, display );
  	const std::string window_name = "matches";
  	cv::namedWindow(window_name, CV_WINDOW_AUTOSIZE);
  	cv::imshow(window_name, display);
		// save images
		string dir_name( "Output/");	
		string file_format( "match_%04d.jpg");
		string path_format= dir_name + file_format;
		char filename[ MAX_FILE_PATH ];
		sprintf( filename, path_format.c_str(), mCallCount );
		cv::imwrite( filename, display );
	}

	// threshold for the door recognition
  double recog_thresh= 0.5;
	// threshold for the duplicated features. There might be many "n to 1" feature matching.
  double duplicate_match_thresh= 3.0; 
  int num_matches= all_indices.size();
  double recog= (double)num_matches/keypoints.size();     // the real number of matches/the number of features in the input image
	double duplicate_match_ratio= (double)matches.size()/num_matches; // the real number of matches/all the number of matches including the duplicated matches
  printf( "number of mathces: %d, recog_ratio= %0.2f, duplicate_match_ratio= %0.2f\n", num_matches, recog, duplicate_match_ratio );
 	mLogFile << "number of mathces: " << num_matches << " recog_ratio= " << recog << "duplicate_ratio= " << duplicate_match_ratio;
  if( recog <= recog_thresh ||  duplicate_match_ratio >= duplicate_match_thresh ) // satisfies both the recognition and duplicate thresholds
  {
	 	printf(" ************* NOT MATCHED *************** \n" );
		mLogFile << "U" << endl;  // U is unmatched
    // exit(0);
		return false;
  }
	mLogFile << "M" << endl;  // M is matched

	// increase the call count of the matchScene function. This variable influences the file name of the matched image.
	mCallCount++;

	return true;
}

/*
 * Compute 3D points of the matched points using stereo geometry.
 * Then, get the mean of the 3D points.
 * @return true if successful
 */
bool HRP2Service::computeTargetPose()
{
	// set up stereo frame
 	Frame frame;
 	frame.setCamParams(mCamParam);        // this sets the projection and reprojection matrices

 	int ndisp= 256; // number of disparities
 	SparseStereo st = SparseStereo( mRectImg0, mRectImg1, true, ndisp );

 	double disp_threshold= 21.0;  // smaller than about 6 meters	
 	Eigen::Vector3d sumTarget( 0.0, 0.0, 0.0 );
 	int num_feasible_matches= 0;
  cout.precision(5);

  std::vector<double> oldpoints_x;
  std::vector<double> oldpoints_y;
  std::vector<double> oldpoints_z;
 	for (int i=0; i<(int)mMatchPoints.size(); i++)
 	{
   	double match_x= mMatchPoints[i].pt.x;
   	double match_y= mMatchPoints[i].pt.y;
		double disp = st.lookup_disparity( match_x, match_y );
		if ( disp > disp_threshold )           // good disparity
    {
		   	Eigen::Vector3d pt( match_x, match_y, disp );
		   	Eigen::Vector3d carTarget = frame.pix2cam(pt);
  	  	sumTarget= sumTarget + carTarget;
 	    	num_feasible_matches++;

				oldpoints_x.push_back( carTarget(0) );
				oldpoints_y.push_back( carTarget(1) );
				oldpoints_z.push_back( carTarget(2) );
			
 	    	printf( "target points (x,y,z)= [%0.3f, %0.3f, %0.3f]\n", carTarget(0), carTarget(1), carTarget(2) );
 	    	mLogFile << "target points (x,y,z)= " << "( " << carTarget(0) << ", " << carTarget(1) << ", " << carTarget(2) << " )" << endl;
		}
	}
	Eigen::Vector3d target;
  target= sumTarget/num_feasible_matches;
  printf( "mean of target point (x,y,z)= [%0.3f, %0.3f, %0.3f]\n", target(0), target(1), target(2) ); 
  mLogFile << "mean of target point (x,y,z)= " << "( " << target(0) << ", " << target(1) << ", " << target(2) << " )" << endl;

	// get the variance of depths to exclude the outliers
	double z_mean= target(2);
	double z_var= getVariance( oldpoints_z, z_mean );
	cout << "mean depth: " << z_mean << " var_depth " << z_var << endl;
	mLogFile << "mean depth: " << z_mean << " var_depth " << z_var << endl;
 
	// get the new mean after removing the outliers
	std::vector<double> newpoints_x;
  std::vector<double> newpoints_y;
  std::vector<double> newpoints_z;
  const double var_coeff= 1.0;      // variance. if sigma=1 then, 68.3% of gaussian distribution, 2 -> 95%
  for( int i=0; i<(int)oldpoints_z.size(); i++ )
	{
		if( (oldpoints_z[i] <= z_mean + var_coeff*z_var ) && 
		    (oldpoints_z[i] >= z_mean - var_coeff*z_var ) )
		{
			newpoints_x.push_back( oldpoints_x[i] );
			newpoints_y.push_back( oldpoints_y[i] );
			newpoints_z.push_back( oldpoints_z[i] );
		}
	}

	// copy the target pose to the member variable
	mTargetPose[0]= getMean(newpoints_x);
	mTargetPose[1]= getMean(newpoints_y);
	mTargetPose[2]= getMean(newpoints_z);
	cout << "after removing outliers, mean [x,y,z]= " << mTargetPose[0] << ", " << mTargetPose[1] << ", " << mTargetPose[2] << endl;		
	mLogFile << "after removing outliers, mean [x,y,z]= " << mTargetPose[0] << ", " << mTargetPose[1] << ", " << mTargetPose[2] << endl;		

 	return true;	
}

static uchar bcolors[][3] = 
{
    {0,0,255},
    {0,128,255},
    {0,255,255},
    {0,255,0},
    {255,128,0},
    {255,255,0},
    {255,0,0},
    {255,0,255},
    {255,255,255}
};

/*
 * Detect the door knob from an image
 * @param aImg image source of Mat type
 * @param win_name window name to disply blobs
 * @return centroid pixel (px, py) of the blob of the knob
 * @Note the real height of a blob cannot be obtained in this stage since disparity is the next step
 * @algorithm
 *  1. segment an image with MSER
 *  2. apply constraints; blob size, blob ratio in order.
 *  3. check the height of the blob
 *  4. If multiple blobs that satisfy the constraints, compute score= w1*blob_size + w2*blob_ratio
 */
Eigen::Vector4d HRP2Service::extractKnob( double aDist2Door, cv::Mat& aImg, const std::string win_name )
{

	cout << endl;
	cout << win_name << endl;
	mLogFile << endl;
	mLogFile << win_name << endl;


	const int THRESH_BLOB_SIZE= 1000;
	const double THRESH_LOWER_WH_RATIO= 3.0; // lower limit for ratio of width over height
	const double THRESH_UPPER_WH_RATIO= 6.0; // upper limit for ratio of width over height
	const double STANDARD_WH_RATIO= 4.8;     // ratio of standard width over height to compute the weight for the good blob
	const double WEIGHT_WH_RATIO= 5000.0;    // weight for the wh ratio

	Eigen::Vector4d g_centroid(0.0, 0.0, 0.0, 0.0);

  // Convert to IplImage or CvMat, no data copying
	IplImage iplImg= aImg; 
 	IplImage* rsp= cvCreateImage( cvGetSize(&iplImg), IPL_DEPTH_8U, 3 );  
	cvCvtColor( &iplImg, rsp, CV_GRAY2BGR );

	CvSeq* contours;
	CvMemStorage* storage= cvCreateMemStorage();
	IplImage* hsv = cvCreateImage( cvGetSize( rsp ), IPL_DEPTH_8U, 3 );
	cvCvtColor( rsp, hsv, CV_BGR2YCrCb );
	CvMSERParams params = cvMSERParams();//cvMSERParams( 5, 60, cvRound(.2*img->width*img->height), .25, .2 );

	double t = (double)cvGetTickCount();
	cvExtractMSER( hsv, NULL, &contours, storage, params );
	t = cvGetTickCount() - t;
	printf( "MSER extracted %d contours in %g ms.\n", contours->total, t/((double)cvGetTickFrequency()*1000.) );

	uchar* rsptr = (uchar*)rsp->imageData;
	// draw MSER with different color
  int blob_id= 0;
	int max_score= 0;

	Eigen::Vector2d centroid(0.0, 0.0);
	// loop through all the blobs (segments)
	for ( int i = contours->total-1; i >= 0; i-- )
	{
		CvSeq* r = *(CvSeq**)cvGetSeqElem( contours, i );
		// first, check the size of a blob
		int blob_size= r->total;
		if( blob_size < THRESH_BLOB_SIZE ) continue; // if the size is smaller than the threshold

		/// get the max and min of width and height as image pixels
		int max_y=0, max_x= 0;
		int min_y= aImg.rows, min_x= aImg.cols, sum_r=0, sum_c=0;
		for( int j=0; j<r->total; j++ )
		{	
			CvPoint* pt = CV_GET_SEQ_ELEM( CvPoint, r, j );
			if( pt->x < min_x ) min_x= pt->x;
			if( pt->x > max_x ) max_x= pt->x;
			if( pt->y < min_y ) min_y= pt->y;
			if( pt->y > max_y ) max_y= pt->y;
      // prepare the sums to get the centroid of the blob
			sum_c+= pt->x;	    
			sum_r+= pt->y;
		}

		// second, check the ratio of width against height
		double width_px= max_x - min_x;
		double height_px= max_y - min_y;
		double wh_ratio= width_px/height_px;
		if( wh_ratio < THRESH_LOWER_WH_RATIO || wh_ratio > THRESH_UPPER_WH_RATIO ) continue;	// if the WH ratio is not in the range

		/// Check the height of the blob
		// get the centroid of the blob
		centroid(0)= double(sum_c)/(r->total);
		centroid(1)= double(sum_r)/(r->total);
		// call the checking function
		double camera_height= mBaseTransformMat[11];
		bool bPass= checkHeightContraint( centroid, camera_height, aDist2Door ); 
		if( !bPass )
		{
			cout << "the height of the blob is not in the range" << endl;
			mLogFile << "the height of the blob is not in the range" << endl;
//			return g_centroid;
		}

		// compute the score of a candidate blob
		double score= blob_size + WEIGHT_WH_RATIO*(1/abs(STANDARD_WH_RATIO-wh_ratio)+1);  // max of the second part is 1 in case of wh_ratio=4
	
		cout.precision(3);
		cout << "blob size, wh_ratio,  score= " << blob_size << ", " << wh_ratio << ", " << score << endl; 
		mLogFile << "blob size, wh_ratio, score= " << blob_size << ", " << wh_ratio << ", " << score << endl; 
    	// update the blob id to choose the most suitable blob		
		if( max_score < score )
		{
			max_score= score;
			blob_id= i;
			g_centroid(0)= centroid(0);    // update the best centroid
			g_centroid(1)= centroid(1);
			g_centroid(2)= width_px;
			g_centroid(3)= height_px;
		}
	}   // end of the loop over all the blobs

	// the best knob
	CvSeq* r = *(CvSeq**)cvGetSeqElem( contours, blob_id );
	cout << "the best knob (blob) is " << r->total << endl;
	mLogFile << "the best knob (blob) is " << r->total << endl;

  // Check whether a blob satisfying the above conditions is detected
	if( g_centroid(0) == 0.0 )
	{
		printf( "*************  KNOB IS NOT DETECTED ***************\n" );
		mLogFile << "************* KNOB IS NOT DETECTED ***************" << endl;
		return g_centroid;
	}

  // Draw the blob of the knob
	if( mVisualizeFlag )
	{
  		CvSeq* knob= *(CvSeq**)cvGetSeqElem( contours, blob_id );
		for( int j=0; j<knob->total; j++ )
		{
			CvPoint* pt = CV_GET_SEQ_ELEM( CvPoint, knob, j );
			rsptr[pt->x*3+pt->y*rsp->widthStep] = bcolors[blob_id%9][2];
			rsptr[pt->x*3+1+pt->y*rsp->widthStep] = bcolors[blob_id%9][1];
			rsptr[pt->x*3+2+pt->y*rsp->widthStep] = bcolors[blob_id%9][0];
		}		
  		cv::namedWindow( win_name );
  		cv::imshow( win_name, rsp );

		// save images
		string dir_name( "Output/");	
		std::string imgfile= win_name + ".png";
		string file_full_name= dir_name + imgfile;
		cv::imwrite( file_full_name, rsp );
	}

  cvReleaseImage( &rsp );  

  return g_centroid;
}

/*
 * Check the height of a blob from the floor if it is in the range.
 * @param aCentroid centroid of blob in the image plane.
 * @param aCameraHeight the height of the camera from the floor.
 * @param aDist2Door distance to the door.
 * @return true if the blob is in the range.
 */
bool HRP2Service::checkHeightContraint( Eigen::Vector2d aCentroid, double aCameraHeight, double aDist2Door )
{
	// get the ratio of meter over pixel
	m_kappa= aDist2Door*tan(V_VIEW_ANGLE/2)/mCamParam.cy;
 	cout << "[checkHeight] kappa= " << m_kappa << endl;
 	mLogFile << "[checkHeight] kappa= " << m_kappa << endl;

	double r_diff= m_kappa*(mCamParam.cy - aCentroid(1));

	double height= aCameraHeight + r_diff;
	cout << "[checkHeight] height= " << height << endl;
	mLogFile << "[checkHeight] height= " << height << endl;

	if(  MIN_HEIGHT <= height && height <= MAX_HEIGHT )   // the knob is in the range
		return true;
	return false;
}

/* 
 * Main function that starts the CORBA connection.
 */
int main(int argc, char** argv)
{
  // --------------------------------------------------------------------------
  // Start CORBA server:
  // --------------------------------------------------------------------------
                                                                                
  try {
    //------------------------------------------------------------------------
    // 1) Initialize ORB
    // 2) Get reference to root POA
    // 3) Bind to name service
    // 4) Initialize servant object
    //------------------------------------------------------------------------
                                                                                
    //------------------------------------------------------------------------
    // Initialize CORBA ORB - "orb"
    //------------------------------------------------------------------------
    CORBA::ORB_var orb = CORBA::ORB_init(argc, argv);
                                                                                
    //------------------------------------------------------------------------
    // Servant must register with POA in order to be made available for client
    // Get reference to the RootPOA.
    //------------------------------------------------------------------------
    CORBA::Object_var obj = orb->resolve_initial_references("RootPOA");
    PortableServer::POA_var _poa = PortableServer::POA::_narrow(obj.in());
                                                                                
    //------------------------------------------------------------------------
    // Operations defined in object interface invoked via an object reference.
    // Instance of CRequestSocketStream_i servant is initialized.
    //------------------------------------------------------------------------
    HRP2Service* myHRP2Service = new HRP2Service();
                                                                                
    //------------------------------------------------------------------------
    // ObjectId_var class defined in poa.h
    // typedef String_var ObjectId_var; CORBA_ORB.h
    // ???????
                                                                                
    //------------------------------------------------------------------------
    // Servant object activated in RootPOA.
    // (Object id used for various POA operations.)
    //------------------------------------------------------------------------
    PortableServer::ObjectId_var myHRP2Service_oid= _poa->activate_object( myHRP2Service );
                                                                                
    //------------------------------------------------------------------------
    // Obtain object reference from servant and register in naming service(??)
    //------------------------------------------------------------------------
    CORBA::Object_var SA_obj = myHRP2Service->_this();
                                                                                
    //------------------------------------------------------------------------
    // Obtain a reference to the object, and print it out as string IOR.
    //------------------------------------------------------------------------
    CORBA::String_var sior( orb->object_to_string( SA_obj.in() ) );
    cout << "'" << (char*)sior << "'" << endl;
                                                                                
    //========================================================================
    // Bind (rebind) object (orb) to name (SA_obj)
    //========================================================================

                                                                      
    //------------------------------------------------------------------------mRect
    // Bind object to name service as defined by directive InitRef
    // and identifier "OmniNameService" in config file omniORB.cfg.
    //------------------------------------------------------------------------
//	cout << "List of Services" << endl;
//	CORBA::ORB::ObjectIdList *idlist = NULL;
//	idlist = orb->list_initial_services();
//	for(int i=0; i<5; i++)
//		cout << (*idlist)[i] << endl;
	CORBA::Object_var obj1= orb->resolve_initial_references("NameService");
	assert( !CORBA::is_nil( obj1.in() ) );
                                         
    //------------------------------------------------------------------------
    // narrow this to the naming context
    //------------------------------------------------------------------------
    CosNaming::NamingContext_var nc = CosNaming::NamingContext::_narrow( obj1.in() );
    assert( !CORBA::is_nil( nc.in() ) );
                                      
                                                                                
    //------------------------------------------------------------------------
    // Bind to CORBA name service. Same name to be requested by client.
    //------------------------------------------------------------------------
    CosNaming::Name name;
    name.length(1);
    name[0].id=CORBA::string_dup("HRP2Vision");
    nc->rebind( name, SA_obj.in() );

                                                                     
    //========================================================================
                                                                                
    myHRP2Service->_remove_ref();
                                                                                
    //------------------------------------------------------------------------
    // Activate the POA manager
    //------------------------------------------------------------------------
    PortableServer::POAManager_var pmgr = _poa->the_POAManager();
    pmgr->activate();
                                                                                
    //------------------------------------------------------------------------
    // Accept requests from clients
    //------------------------------------------------------------------------
    orb->run();
                                                                                
    //------------------------------------------------------------------------
    // If orb leaves event handling loop.
    // - currently configured never to time out (??)
    //------------------------------------------------------------------------
    orb->destroy();
                                                                                
    free(name[0].id); // str_dup does a malloc internally
  }
 
 catch( const CosNaming::NamingContext::NotFound & )
 {
	cerr << "No name for " << endl;
 }
                                                          
  catch(CORBA::SystemException&) {
    cerr << "Caught CORBA::SystemException." << endl;
  }
  catch(CORBA::Exception&) {
    cerr << "Caught CORBA::Exception." << endl;
  }
  catch(omniORB::fatalException& fe) {
    cerr << "Caught omniORB::fatalException:" << endl;
    cerr << "  file: " << fe.file() << endl;
    cerr << "  line: " << fe.line() << endl;
    cerr << "  mesg: " << fe.errmsg() << endl;
  }
  catch(...) {
    cerr << "Caught unknown exception." << endl;
  }
                                                                                
  return 0;
}
