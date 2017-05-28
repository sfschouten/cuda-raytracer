// OpenGL Graphics includes
#include <helper_gl.h>
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <GL/wglew.h>
#endif
#if defined(__APPLE__) || defined(__MACOSX)
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// CUDA runtime
// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>

#include "Raytracer_kernel.h"

#include "Sphere.h"
#include "Plane.h"

#define MAX_EPSILON_ERROR 5.0f

FILE *stream ;
char g_ExecPath[300] ;

// Set to 1 to run on the CPU instead of the GPU for timing comparison.
#define RUN_CPU 0

// Set to 1 to time frame generation
#define RUN_TIMING 0

// Random number macros
#define RANDOMSEED(seed) ((seed) = ((seed) * 1103515245 + 12345))
#define RANDOMBITS(seed, bits) ((unsigned int)RANDOMSEED(seed) >> (32 - (bits)))

//OpenGL PBO and texture "names"
GLuint gl_PBO, gl_Tex, gl_Shader;
struct cudaGraphicsResource *cuda_pbo_resource; // handles OpenGL-CUDA exchange

//Source image on the host side
uchar4 *h_Src = 0;

// Destination image on the GPU side
uchar4 *d_Dst = NULL;

//Initial image width and height
int imageW = 520, imageH = 520;

// Timer ID
StopWatchInterface *hTimer = NULL;

// User interface variables
int lastx = 0;
int lasty = 0;
bool leftClicked = false;
bool middleClicked = false;
bool rightClicked = false;

int numSMs = 0;          // number of multiprocessors
int version = 1;         // Compute Capability

// Auto-Verification Code
const int frameCheckNumber = 60;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 15;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;

int *pArgc = NULL;
char **pArgv = NULL;


//Raytracer
Camera *h_Camera;
Camera *d_Camera;
Scene *h_Scene;
Scene *d_Scene;
Vector3 *h_Directions;
Vector3 *d_Directions;


#define MAX_EPSILON 50
#define REFRESH_DELAY     10 //ms

#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif
#define BUFFER_DATA(i) ((char *)0 + i)

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
// This is specifically to enable the application to enable/disable vsync
typedef BOOL (WINAPI *PFNWGLSWAPINTERVALFARPROC)(int);

void setVSync(int interval)
{
    if (WGL_EXT_swap_control)
    {
        wglSwapIntervalEXT = (PFNWGLSWAPINTERVALFARPROC)wglGetProcAddress("wglSwapIntervalEXT");
        wglSwapIntervalEXT(interval);
    }
}
#endif

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&hTimer) / 1000.f);
        sprintf(fps, "<CUDA Raytracer> %3.1f fps", ifps);
        glutSetWindowTitle(fps);
        fpsCount = 0;

        fpsLimit = MAX(1.f, (float)ifps);
        sdkResetTimer(&hTimer);
    }
}

void printCudaLastError()
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n", __FILE__, __LINE__, "", (int)err, cudaGetErrorString(err));
}

// render Mandelbrot image using CUDA or CPU
void renderImage()
{
	float timeEstimate;
	sdkResetTimer(&hTimer);

	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_Dst, &num_bytes, cuda_pbo_resource));

	//Copy arguments to device
	d_Camera = h_Camera->copyToDevice();
	d_Scene = h_Scene->copyToDevice();

	printCudaLastError();

	//Run raytracer on device.
	RunRaytrace(d_Dst, imageW, imageH, d_Camera, d_Directions, d_Scene);

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
	
	cudaDeviceSynchronize();
	printCudaLastError();
}

// OpenGL display function
void displayFunc(void)
{
    sdkStartTimer(&hTimer);

    // render the image
    renderImage();

    glBindTexture(GL_TEXTURE_2D, gl_Tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageW, imageH, GL_RGBA, GL_UNSIGNED_BYTE, BUFFER_DATA(0));
    glDisable(GL_DEPTH_TEST);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(0.0f, 0.0f);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(1.0f, 0.0f);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(0.0f, 1.0f);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);

    sdkStopTimer(&hTimer);
    glutSwapBuffers();

    computeFPS();
}

void cleanup()
{
    if (h_Src)
    {
        free(h_Src);
        h_Src = 0;
    }
	if (h_Directions)
	{
		free(h_Directions);
		h_Directions = 0;
	}

    sdkStopTimer(&hTimer);
    sdkDeleteTimer(&hTimer);

    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    glDeleteBuffers(1, &gl_PBO);
    glDeleteTextures(1, &gl_Tex);
    glDeleteProgramsARB(1, &gl_Shader);
}

void initMenus();

// OpenGL keyboard function
void keyboardFunc(unsigned char k, int, int)
{
    int seed;

    switch (k)
    {
        case '\033':
        case 'q':
        case 'Q':
            printf("Shutting down...\n");

            #if defined(__APPLE__) || defined(MACOSX)
            exit(EXIT_SUCCESS);
            #else
            glutDestroyWindow(glutGetWindow());
            return;
            #endif
            break;
		case 'a':
        case '4':   // Left arrow key
            
            break;
		case 'w':
        case '8':   // Up arrow key
            
            break;
		case 'd':
        case '6':   // Right arrow key
            
            break;
		case 's':
        case '2':   // Down arrow key
            
            break;

        case '+':
            
            break;

        case '-':
            
            break;

        default:
            break;
    }

} // keyboardFunc

// OpenGL mouse click function
void clickFunc(int button, int state, int x, int y)
{
    if (button == 0)
        leftClicked = !leftClicked;

    if (button == 1)
        middleClicked = !middleClicked;

    if (button == 2)
        rightClicked = !rightClicked;

    int modifiers = glutGetModifiers();

    if (leftClicked && (modifiers & GLUT_ACTIVE_SHIFT))
    {
        leftClicked = 0;
        middleClicked = 1;
    }

    if (state == GLUT_UP)
    {
        leftClicked = 0;
        middleClicked = 0;
    }

    lastx = x;
    lasty = y;
} // clickFunc

// OpenGL mouse motion function
void motionFunc(int x, int y)
{
    double fx = (double)(x - lastx) / 50.0 / (double)(imageW);
    double fy = (double)(lasty - y) / 50.0 / (double)(imageH);

    if (leftClicked)
    {
        
    }
    else
    {
        
    }

    if (middleClicked)
        if (fy > 0.0f)
        {
            
        }
        else
        {
            
        }
    else
    {
        
    }
} // motionFunc

void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    }
}

void mainMenu(int i)
{
    
}

void initMenus()
{
    glutCreateMenu(mainMenu);

    glutAttachMenu(GLUT_RIGHT_BUTTON);
}

void initOpenGLBuffers(int w, int h)
{
    // delete old buffers
    if (h_Src)
    {
        free(h_Src);
        h_Src = 0;
    }
    if (gl_Tex)
    {
        glDeleteTextures(1, &gl_Tex);
        gl_Tex = 0;
    }
    if (gl_PBO)
    {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &gl_PBO);
        gl_PBO = 0;
    }

    // check for minimized window
    if ((w==0) && (h==0)) 
        return; 

    // allocate new buffers
    h_Src = (uchar4 *)malloc(w * h * 4);

    printf("Creating GL texture...\n");
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &gl_Tex);
    glBindTexture(GL_TEXTURE_2D, gl_Tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_Src);
    printf("Texture created.\n");

    printf("Creating PBO...\n");
    glGenBuffers(1, &gl_PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, w * h * 4, h_Src, GL_STREAM_COPY);
    //While a PBO is registered to CUDA, it can't be used
    //as the destination for OpenGL drawing calls.
    //But in our particular case OpenGL is only used
    //to display the content of the PBO, specified by CUDA kernels,
    //so we need to register/unregister it only once.

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO, cudaGraphicsMapFlagsWriteDiscard));
    printf("PBO created.\n");
}

void initDirections()
{
	//Clear Directions.
	if (h_Directions)
	{
		free(h_Directions);
		h_Directions = 0;
	}

	//Allocate memory for directions.
	int directions_size = imageW * imageH * sizeof(Vector3);
	h_Directions = new Vector3[imageW * imageH];

	//Fill directions.
	for (int y = 0; y < imageH; y++)
		for (int x = 0; x < imageW; x++)
			h_Directions[y * imageW + x] = h_Camera->getPixelDirection(x, y, imageW, imageH);

	//Copy directions to device.
	cudaMalloc((void **)&d_Directions, directions_size);
	cudaMemcpy(d_Directions, h_Directions, directions_size, cudaMemcpyHostToDevice);
}

void initRaytracing()
{ 
	if (h_Camera)
		delete h_Camera;

	h_Camera = new Camera(Vector3(5, 1, 0), Vector3(5, 1, 5));
	h_Camera->update();

	if (h_Scene)
		delete h_Scene;

	h_Scene = new Scene(Primitive(Material(make_float3(0, 0, 0), make_float3(0, 0, 0))));

	h_Scene->addLight(new Light(Vector3(3, 0, 0), make_float3(8, 8, 8)));
	h_Scene->addLight(new Light(Vector3(7, 2, 0), make_float3(8, 8, 8)));

	h_Scene->addSphere(new Sphere(1, Vector3(2.8f, 1, 5), Material(make_float3(0.5, 0, 0), make_float3(0.5, 0.5, 0.5))));
	h_Scene->addSphere(new Sphere(1, Vector3(5, 1, 4), Material(make_float3(0, 0.2, 0), make_float3(0.8, 0.8, 0.8))));
	h_Scene->addSphere(new Sphere(1, Vector3(7.2f, 1, 5), Material(make_float3(0, 0, 1), make_float3(0, 0, 0))));

	initDirections();
}

void reshapeFunc(int w, int h)
{
    glViewport(0, 0, w, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    initOpenGLBuffers(w, h);
    imageW = w;
    imageH = h;

	initDirections();
}

void initGL(int *argc, char **argv)
{
    printf("Initializing GLUT...\n");
    glutInit(argc, argv);

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(imageW, imageH);
    glutInitWindowPosition(0, 50);
    glutCreateWindow(argv[0]);

    glutDisplayFunc(displayFunc);
    glutKeyboardFunc(keyboardFunc);
    glutMouseFunc(clickFunc);
    glutMotionFunc(motionFunc);
    glutReshapeFunc(reshapeFunc);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    initMenus();

    if (!isGLVersionSupported(1,5) ||
        !areGLExtensionsSupported("GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object"))
    {
        fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
        fprintf(stderr, "This sample requires:\n");
        fprintf(stderr, "  OpenGL version 1.5\n");
        fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
        fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
        exit(EXIT_SUCCESS);
    }

    printf("OpenGL window created.\n");
}

void initData(int argc, char **argv)
{
    // check for hardware double precision support
    int dev = 0;
    dev = findCudaDevice(argc, (const char **)argv);

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
    version = deviceProp.major*10 + deviceProp.minor;

    if (version < 11)
    {
        printf("GPU compute capability is too low (1.0), program is waived\n");
        exit(EXIT_WAIVED);
    }

    numSMs = deviceProp.multiProcessorCount;

	initRaytracing();
}

// General initialization call for CUDA Device
void chooseCudaDevice(int argc, const char **argv, bool bUseOpenGL)
{
    if (bUseOpenGL)
        findCudaGLDevice(argc, argv);
    else
        findCudaDevice(argc, argv);
}

void printHelp()
{
    printf("[Mandelbrot]\n");
    printf("\tUsage Parameters\n");
    printf("\t-device=n        (requires to be in non-graphics mode)\n");
    printf("\t-file=output.ppm (output file for image testing)\n");
    printf("\t-mode=0,1        (0=Mandelbrot Set, 1=Julia Set)\n");
    printf("\t-fp64            (run in double precision mode)\n");
}


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    pArgc = &argc;
    pArgv = argv;

#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

    printf("Starting...\n");

    // parse command line arguments
    if (checkCmdLineFlag(argc, (const char **)argv, "help"))
    {
        printHelp();
        exit(EXIT_SUCCESS);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "file"))
    {
        fpsLimit = frameCheckNumber;

        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        findCudaDevice(argc, (const char **)argv); // no OpenGL usage

        // If the GPU does not meet SM1.1 capabilities, we will quit
        if (!checkCudaCapabilities(1,1)) 
            exit(EXIT_SUCCESS); 

        exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
    }
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    else if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        printf("[%s]\n", argv[0]);
        printf("   Does not explicitly support -device=n in OpenGL mode\n");
        printf("   To use -device=n, the sample must be running w/o OpenGL\n\n");
        printf(" > %s -device=n -file=<image_name>.ppm\n", argv[0]);
        printf("exiting...\n");
        exit(EXIT_SUCCESS);
    }

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    chooseCudaDevice(argc, (const char **)argv, true); // yes to OpenGL usage

    // If the GPU does not meet SM1.1 capabilities, we quit
    if (!checkCudaCapabilities(1,1))
        exit(EXIT_SUCCESS);

    // Otherwise it succeeds, we will continue to run this sample
    initData(argc, argv);

    // Initialize OpenGL context first before the CUDA context is created.  This is needed
    // to achieve optimal performance with OpenGL/CUDA interop.
    initGL(&argc, argv);
    initOpenGLBuffers(imageW, imageH);

	//cudaDeviceSetLimit(cudaLimitMallocHeapSize, 268435456);
	cudaDeviceSetLimit(cudaLimitStackSize, 16 * 1024);

	size_t heapSize;
	cudaDeviceGetLimit(&heapSize, cudaLimitMallocHeapSize);
	printf("Heap size: %zd\n", heapSize);

	size_t stackSize;
	cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
	printf("Stack size: %zd\n", stackSize);

    printf("Starting GLUT main loop...\n");
    printf("\n");
    printf("Press [e] to reset\n");
    printf("Right mouse button = Menu\n");
    printf("Press [q] to exit\n");
    printf("\n");

    sdkCreateTimer(&hTimer);
    sdkStartTimer(&hTimer);

#if defined (__APPLE__) || defined(MACOSX)
    atexit(cleanup);
#else
    glutCloseFunc(cleanup);
#endif

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    setVSync(0) ;
#endif

    glutMainLoop();
} // main
