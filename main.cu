
// Define Defaults ------------------------------------------------------------
#define KERNEL_GEOMETRY_NUM_BLOCKS				48
#define KERNEL_GEOMETRY_NUM_THREADS_PER_BLOCK 	256
#define KERNEL_RASTERIZE_NUM_BLOCKS				16
#define KERNEL_RASTERIZE_NUM_THREADS_PER_BLOCK 	128

// includes -------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <time.h>

#include "structures.c"
#include "helpers.c"
#include "helpers.cu"
#include "kernel_geometry.cu"	
#include "kernel_rasterizer.cu"	

const int SCREEN_HEIGHT = 300;
const int SCREEN_WIDTH = 300;
const int TEX_WIDTH = 300;
const int TILE_SIZE = 20;
const int COLOR_DEPTH = 256;

const float LEFT = -1.0;
const float RIGHT = 1.0;
const float BOTTOM = -1.0;
const float TOP = 1.0;
const float NEAR = 1.0;
const float FAR = 4.0;
				
const float ORIGIN_X = 0.0;
const float ORIGIN_Y = 0.0;
const float ORIGIN_Z = 1.5;				
						
int buildArrayMeshSphere( float radius, int divisions, int tex_width, float x, float y, float z, 
							Vertex4D **baseVerts, Color4D **baseColors, Vertex3D **baseNormals, Vertex3D **baseTCs );
int assembleTrianglesSphere( int numVerts, int divisions, Vertex4D *v_arr, Vertex3D *n_arr, Color4D *c_arr, Vertex3D *t_arr, Triangle **tri_arr );

void runSerialPipe( AppConfig ap, float radius, int divisions );
void runCudaPipe( AppConfig ap, float radius, int divisions );
void areaTest( AppConfig ap );
void geometryTest( AppConfig ap );

/* ////////////////////////////////////////
// Methods //
//////////////////////////////////////// */

int main( int argc, char **argv )
{
	AppConfig ap = initialize(argc, argv);
	
	printAppConfig( ap );
	
	if( ap.do_area_test ) areaTest( ap );
	else if( ap.do_geom_test ) geometryTest( ap );
	else{
		if(ap.do_parallel) runCudaPipe( ap, ap.radius, ap.divisions );
		else runSerialPipe( ap, ap.radius, ap.divisions );
	}

	exit(0);
}

void areaTest( AppConfig ap ) 
{
	int i;
	float radius = 0.1;
	printf( "Running Area Test\n" );
	printf( "General\t\t\t\t\tGeometry\t\t\tRasterization\n" );
	printf( "Radius\tDivs\tArea\tTris\t\tDrawPct\tTime(s)\tRate(T/s)\tDrawPct\tTime(s)\tRate(T/s)\n" );
	for(i=0; i<30; i++){
		runCudaPipe( ap, radius, ap.divisions );
		radius += 0.05;
	}
}

void geometryTest( AppConfig ap )
{
	int i, divisions = 1;
	printf( "Running Geometry Test\n" );
	printf( "General\t\t\t\t\tGeometry\t\t\tRasterization\n" );
	printf( "Radius\tDivs\tArea\tTris\t\tDrawPct\tTime(s)\tRate(T/s)\tDrawPct\tTime(s)\tRate(T/s)\n" );
	for(i=0; i<30; i++){
		runCudaPipe( ap, ap.radius, divisions );
		divisions+=10;		
	}
}

void runCudaPipe( AppConfig ap, float radius, int divisions ) 
{ 
	int i, j;
	int num_vertices, num_triangles;
	cudaEvent_t start, stop;
	float time_geom = 0.0, time_rast = 0.0;
	float draw_pct_geom = 0.0, draw_pct_rast = 0.0;
	int drawn_count_geom = 0, drawn_count_rast = 0;
	
	Vertex4D *baseVerts;
	Color4D *baseColors;
	Color4D *tex_buffer_h, *tex_buffer_d;
	Color4D *frame_buffer_h, *frame_buffer_d;
	Vertex3D *baseNormals, *baseTCs;
	Triangle *triangles_h, *triangles_d;
	Float1D *depth_buffer_h, *depth_buffer_d;
	Matrix4D worldSpace4D, cameraSpace4D, translate4D, scale4D;
	Matrix4D modelView4D, projection4D, viewport4D;
	
	/* construct the triangle sphere */
	num_vertices = buildArrayMeshSphere( radius, divisions, TEX_WIDTH, ORIGIN_X, ORIGIN_Y, ORIGIN_Z, 
											&baseVerts, &baseColors, &baseNormals, &baseTCs );
	num_triangles = assembleTrianglesSphere( num_vertices, divisions, baseVerts, baseNormals, baseColors, baseTCs, &triangles_h );
	
	/* allocate and initialize accessory arrays on host */
	frame_buffer_h = (Color4D *)malloc(sizeof(Color4D)*SCREEN_HEIGHT*SCREEN_WIDTH);
	depth_buffer_h = (Float1D *)malloc(sizeof(Float1D)*SCREEN_HEIGHT*SCREEN_WIDTH);

	buildCheckerBoard( TEX_WIDTH, TILE_SIZE, &tex_buffer_h );

	Int1D *draw_bit_geom_d, *draw_bit_rast_d;
	Int1D *draw_bit_geom_h = (Int1D *)malloc(sizeof(Int1D)*num_triangles);
	Int1D *draw_bit_rast_h = (Int1D *)malloc(sizeof(Int1D)*num_triangles);

	/* initialize buffers */
	for(i=0; i<num_triangles; i++){ 
		draw_bit_geom_h[i].i = 0; 
		draw_bit_rast_h[i].i = 0; 
	}
	int indx;
	for(i=0; i<SCREEN_HEIGHT; i++){
		for(j=0; j<SCREEN_WIDTH; j++){
			indx = j+i*SCREEN_HEIGHT;
			frame_buffer_h[indx].r = 255.0;
			frame_buffer_h[indx].g = 255.0;
			frame_buffer_h[indx].b = 255.0;
			frame_buffer_h[indx].a = 1.0;
			
			depth_buffer_h[indx].f = 100.0;
		}
	}
		
	/* allocate arrays on device, and copy memory from host */
	cudaMalloc((void **) &triangles_d, sizeof(Triangle)*num_triangles);
	
	cudaMalloc((void **) &draw_bit_geom_d, sizeof(Int1D)*num_triangles);
	cudaMalloc((void **) &draw_bit_rast_d, sizeof(Int1D)*num_triangles);
	
	cudaMemcpy(triangles_d, triangles_h, sizeof(Triangle)*num_triangles, cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy: triangles, host to device");

	cudaMemcpy(draw_bit_geom_d, draw_bit_geom_h, sizeof(Int1D)*num_triangles, cudaMemcpyHostToDevice);	
	checkCUDAError("cudaMemcpy: draw bit, host to device");		
	cudaMemcpy(draw_bit_rast_d, draw_bit_rast_h, sizeof(Int1D)*num_triangles, cudaMemcpyHostToDevice);	
	checkCUDAError("cudaMemcpy: draw bit, host to device");		
	
	/*//////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////// PHASE 1: GEOMETRY /////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////*/
	
	worldSpace4D = loadIdentity4D();
	cameraSpace4D = loadIdentity4D();
	modelView4D = mulMatrix4D( cameraSpace4D, worldSpace4D );

	if( ap.do_perspective_correct ) projection4D = projection_perspective( RIGHT, LEFT, BOTTOM, TOP, NEAR, FAR );
	else projection4D = projection_orthographic( RIGHT, LEFT, BOTTOM, TOP, NEAR, FAR );	

	translate4D = translate( (RIGHT-LEFT)/2.0, (TOP-BOTTOM)/2.0, 0.0 );
	scale4D = scale( SCREEN_WIDTH/(RIGHT-LEFT), SCREEN_HEIGHT/(TOP-BOTTOM), 1.0 );
	viewport4D = mulMatrix4D( scale4D, translate4D );
	
	cudaEventCreate(&start);
	checkCUDAError("event create - start: kernel_geometry");	
	cudaEventCreate(&stop);
	checkCUDAError("event create - stop: kernel_geometry");	
	cudaEventRecord(start, 0);	
	checkCUDAError("initial event record: kernel_geometry");	

	kernel_geometry<<< KERNEL_GEOMETRY_NUM_BLOCKS, KERNEL_GEOMETRY_NUM_THREADS_PER_BLOCK >>>( num_triangles, triangles_d, modelView4D, projection4D, 
																								viewport4D, ap.do_perspective_correct, draw_bit_geom_d );
	checkCUDAError("kernel invocation: kernel_geometry");
	
	cudaEventRecord(stop, 0);
	checkCUDAError("final event record: kernel_geometry");
	cudaEventSynchronize(stop);
	checkCUDAError("event sync: kernel_geometry");
	cudaThreadSynchronize();
	checkCUDAError("thread sync: kernel_geometry");
	cudaEventElapsedTime(&time_geom, start, stop);
	checkCUDAError("elapsed time: kernel_geometry");

	cudaMemcpy(draw_bit_geom_h, draw_bit_geom_d, sizeof(Int1D)*num_triangles, cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy: frame buffer, device to host");		

	drawn_count_geom = 0;
	for(i=0; i<num_triangles; i++){ if( draw_bit_geom_h[i].i == 1 ) drawn_count_geom++; }
		
	/*//////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////// PHASE 2: RASTERIZATION ///////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////*/	

	cudaMalloc((void **) &frame_buffer_d, sizeof(Color4D)*SCREEN_HEIGHT*SCREEN_WIDTH);
	cudaMalloc((void **) &depth_buffer_d, sizeof(Float1D)*SCREEN_HEIGHT*SCREEN_WIDTH);
	cudaMalloc((void **) &tex_buffer_d, sizeof(Color4D)*TEX_WIDTH*TEX_WIDTH);

	cudaMemcpy(frame_buffer_d, frame_buffer_h, sizeof(Color4D)*SCREEN_HEIGHT*SCREEN_WIDTH, cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy: frame buffer, host to device");
	cudaMemcpy(depth_buffer_d, depth_buffer_h, sizeof(Float1D)*SCREEN_HEIGHT*SCREEN_WIDTH, cudaMemcpyHostToDevice);	
	checkCUDAError("cudaMemcpy: depth buffer, host to device");
	cudaMemcpy(tex_buffer_d, tex_buffer_h, sizeof(Color4D)*TEX_WIDTH*TEX_WIDTH, cudaMemcpyHostToDevice);	
	checkCUDAError("cudaMemcpy: depth buffer, host to device");
	
	cudaEventCreate(&start);
	checkCUDAError("event create - start: kernel_rasterizer");	
	cudaEventCreate(&stop);
	checkCUDAError("event create - stop: kernel_rasterizer");	
	cudaEventRecord(start, 0);	
	checkCUDAError("initial event record: kernel_rasterizer");	
	
	kernel_rasterizer<<< KERNEL_RASTERIZE_NUM_BLOCKS, KERNEL_RASTERIZE_NUM_THREADS_PER_BLOCK >>>( num_triangles, triangles_d, frame_buffer_d, depth_buffer_d, 									tex_buffer_d, SCREEN_WIDTH, TEX_WIDTH, ap.do_texture, ap.do_perspective_correct, ap.do_depth_test, draw_bit_rast_d );
	checkCUDAError("kernel invocation: kernel_rasterizer");
	
	cudaEventRecord(stop, 0);
	checkCUDAError("final event record: kernel_rasterizer");
	cudaEventSynchronize(stop);
	checkCUDAError("event sync: kernel_rasterizer");
	cudaThreadSynchronize();
	checkCUDAError("thread sync: kernel_rasterizer");
	cudaEventElapsedTime(&time_rast, start, stop);
	checkCUDAError("elapsed time: kernel_rasterizer");

	cudaMemcpy(frame_buffer_h, frame_buffer_d, sizeof(Color4D)*SCREEN_HEIGHT*SCREEN_WIDTH, cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy: frame buffer, device to host");	
	
	cudaMemcpy(draw_bit_rast_h, draw_bit_rast_d, sizeof(Int1D)*num_triangles, cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy: frame buffer, device to host");		

	drawn_count_rast = 0;
	for(i=0; i<num_triangles; i++){ if( draw_bit_rast_h[i].i == 1 ) drawn_count_rast++; }
	
	 draw_pct_geom = ((float)drawn_count_geom)/((float)num_triangles);
	 draw_pct_rast = ((float)drawn_count_rast)/((float)num_triangles);
	
	 printf( "%.2f\t%d\t%.3f\t%d\t\t%.1f\t%.3f\t%.2f\t\t%.1f\t%.3f\t%.2f\n", radius, divisions, (radius*radius*PI), num_triangles, 
		 100*draw_pct_geom, time_geom, ((float)drawn_count_geom*draw_pct_geom)/time_geom, 
		 100*draw_pct_rast, time_rast, ((float)drawn_count_rast*draw_pct_rast)/time_rast );
	
	 if( !ap.do_area_test && !ap.do_geom_test ) writePPMFile( ap.output_file, frame_buffer_h, SCREEN_WIDTH, SCREEN_HEIGHT, COLOR_DEPTH);

	/*//////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////// END PIPE /////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////*/		
	
	//clean up memory
	free((void *)baseVerts);
	free((void *)baseColors);
	free((void *)baseNormals);
	free((void *)baseTCs);
	
	free((void *)frame_buffer_h);
	free((void *)depth_buffer_h);
	free((void *)triangles_h);
	
	cudaFree(frame_buffer_d);
	cudaFree(depth_buffer_d);
	cudaFree(triangles_d);	
}

void runSerialPipe( AppConfig ap, float radius, int divisions )
{
	Vertex4D *baseVerts;
	Color4D *baseColors;
	Vertex3D *baseNormals, *baseTCs;
	Triangle *in_tri;
	Color4D *frame_buffer_d, *tex_buffer_d;
	Float1D* depth_buffer_d;
	Matrix4D translate4D, scale4D, projection4D, viewport4D;

	int numVerts, numTris;
	int i, j, indx;

	frame_buffer_d = (Color4D*)malloc(sizeof(Color4D)*SCREEN_WIDTH*SCREEN_HEIGHT);
	tex_buffer_d = (Color4D*)malloc(sizeof(Color4D)*TEX_WIDTH*TEX_WIDTH);	
	depth_buffer_d = (Float1D*)malloc(sizeof(Float1D)*SCREEN_WIDTH*SCREEN_HEIGHT);
	
	buildCheckerBoard( 300, 20, &tex_buffer_d );
	for(i=0; i<300; i++){
		for(j=0; j<300; j++){
			indx = j+i*300;
			frame_buffer_d[indx].r = 255.0;
			frame_buffer_d[indx].g = 255.0;
			frame_buffer_d[indx].b = 255.0;
			frame_buffer_d[indx].a = 1.0;
			
			depth_buffer_d[indx].f = 100.0;
		}
	}

	translate4D = translate( (RIGHT-LEFT)/2.0, (TOP-BOTTOM)/2.0, 0.0 );
	scale4D = scale( SCREEN_WIDTH/(RIGHT-LEFT), SCREEN_HEIGHT/(TOP-BOTTOM), 1.0 );
	
	translate4D = translate(1.0, 1.0, 0.0);
	scale4D = scale(SCREEN_WIDTH/2.0, SCREEN_HEIGHT/2.0, 1.0);
	
	if( ap.do_perspective_correct ) projection4D = projection_perspective( RIGHT, LEFT, BOTTOM, TOP, NEAR, FAR );
	else projection4D = projection_orthographic( RIGHT, LEFT, BOTTOM, TOP, NEAR, FAR );	
	
	viewport4D = mulMatrix4D( scale4D, translate4D  );	

	numVerts = buildArrayMeshSphere( ap.radius, ap.divisions, TEX_WIDTH, ORIGIN_X, ORIGIN_Y, ORIGIN_Z, 
										&baseVerts, &baseColors, &baseNormals, &baseTCs );
	numTris = assembleTrianglesSphere( numVerts, ap.divisions, baseVerts, baseNormals, baseColors, baseTCs, &in_tri );

	/*//////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////// PHASE 1: GEOMETRY /////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////*/
	
	for(i=0; i<numTris; i++)
	{
		in_tri[i].v[0] = mulMatrix4DVertex4D( projection4D, in_tri[i].v[0] ); 
		in_tri[i].v[1] = mulMatrix4DVertex4D( projection4D, in_tri[i].v[1] );
		in_tri[i].v[2] = mulMatrix4DVertex4D( projection4D, in_tri[i].v[2] );
		
		if( ap.do_perspective_correct ){
			in_tri[i].t[0] = scaleVertex3D( in_tri[i].t[0], 1/in_tri[i].v[0].w ); 
			in_tri[i].t[1] = scaleVertex3D( in_tri[i].t[1], 1/in_tri[i].v[1].w );
			in_tri[i].t[2] = scaleVertex3D( in_tri[i].t[2], 1/in_tri[i].v[2].w );
		}
		
		in_tri[i].v[0] = normalizeW4D( in_tri[i].v[0] ); 
		in_tri[i].v[1] = normalizeW4D( in_tri[i].v[1] );
		in_tri[i].v[2] = normalizeW4D( in_tri[i].v[2] );
		
		in_tri[i].v[0] = mulMatrix4DVertex4D( viewport4D, in_tri[i].v[0] ); 
		in_tri[i].v[1] = mulMatrix4DVertex4D( viewport4D, in_tri[i].v[1] );
		in_tri[i].v[2] = mulMatrix4DVertex4D( viewport4D, in_tri[i].v[2] );
	}

	/*//////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////// PHASE 2: RASTERIZATION ///////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////*/		
	
	for(i=0; i<numTris; i++)
	{	
		fineRast( &in_tri[i], frame_buffer_d, depth_buffer_d, tex_buffer_d, SCREEN_WIDTH, SCREEN_HEIGHT, 
					ap.do_texture, ap.do_perspective_correct, ap.do_depth_test );
	}

	/*//////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////// END PIPE /////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////*/	
	
	writePPMFile( ap.output_file, frame_buffer_d, SCREEN_WIDTH, SCREEN_HEIGHT, COLOR_DEPTH );
	
	free((void *)baseVerts);
	free((void *)baseColors);
	free((void *)baseNormals);
	free((void *)baseTCs);
	
	free((void *)in_tri);
	free((void *)frame_buffer_d);
	free((void *)depth_buffer_d);
	free((void *)tex_buffer_d);
}

int buildArrayMeshSphere( float radius, int divisions, int tex_width, float x, float y, float z, 
							Vertex4D **baseVerts, Color4D **baseColors, Vertex3D **baseNormals, Vertex3D **baseTCs )
{
	int i, j, indx=0;
	Vertex3D *bt;
	Vertex3D *bn;
	Vertex4D *bv;
	Color4D *bc;
	
	int numRows = 2*divisions + 3;
	int numVerts = 2*2*divisions*numRows;
	
	float dTheta = 180.0/(divisions+1);
	float dPhi = 90.0/divisions;
	float theta = 0.0;
	float phi = 0.0;
	float tex_off = (float)tex_width/2.0;
	float tex_mult = tex_off/radius;
	
	bv = *baseVerts = (Vertex4D *)malloc(sizeof(Vertex4D)*numVerts);
	bc = *baseColors = (Color4D *)malloc(sizeof(Color4D)*numVerts);
	bn = *baseNormals = (Vertex3D *)malloc(sizeof(Vertex3D)*numVerts);
	bt = *baseTCs = (Vertex3D *)malloc(sizeof(Vertex3D)*numVerts);
	
	for(i=0; i<2*divisions; i++)
	{	
		theta = 0.0;
	
		for(j=0; j<numRows; j++)
		{	
			bv[indx].x = x + radius*dCos(theta)*dSin(phi);
			bv[indx].y = y + radius*dSin(theta)*dSin(phi);
			bv[indx].z = z + radius*dCos(phi);
			bv[indx].w = 1.0;
			
			bn[indx].x = 0.0;
			bn[indx].y = 0.0;
			bn[indx].z = 0.0;
			
			bc[indx].r = 255*dCos(theta);
			bc[indx].g = 255*dSin(theta);
			bc[indx].b = 0.0;
			bc[indx].a = 1.0;

			bt[indx].x = (x + radius*dCos(theta))*tex_mult + tex_off;
			bt[indx].y = (y + radius*dSin(theta))*tex_mult + tex_off;			
			bt[indx].z = 1.0;
			
			indx++;
			
			bv[indx].x = x + radius*dCos(theta)*dSin(phi+dPhi);
			bv[indx].y = y + radius*dSin(theta)*dSin(phi+dPhi);
			bv[indx].z = z + radius*dCos(phi+dPhi);
			bv[indx].w = 1.0;

			bn[indx].x = 0.0;
			bn[indx].y = 0.0;
			bn[indx].z = 0.0;
			
			bc[indx].r = 255*dCos(theta);
			bc[indx].g = 255*dSin(theta);
			bc[indx].b = 0.0;
			bc[indx].a = 1.0;

			bt[indx].x = (x + radius*dCos(theta))*tex_mult + tex_off;
			bt[indx].y = (y + radius*dSin(theta))*tex_mult + tex_off;	
			bt[indx].z = 1.0;
			
			indx++;
			
			theta += dTheta;
		}
		phi += dPhi;
	}
	
	return numVerts;
}

int assembleTrianglesSphere( int numVerts, int divisions, Vertex4D *v_arr, Vertex3D *n_arr, Color4D *c_arr, Vertex3D *t_arr, Triangle **tri_arr )
{
	int i, idx = 0;
	Triangle *out_arr;
	int numTriangles = numVerts-2;
	int iterations = numTriangles/2;

	out_arr = *tri_arr = (Triangle *)malloc(sizeof(Triangle)*(numTriangles));
	
	for(i=0; i<iterations; i++)
	{
		out_arr[idx].v[0] = v_arr[2*i+0];
		out_arr[idx].v[1] = v_arr[2*i+1];
		out_arr[idx].v[2] = v_arr[2*i+2];
		out_arr[idx].n[0] = n_arr[2*i+0];
		out_arr[idx].n[1] = n_arr[2*i+1];
		out_arr[idx].n[2] = n_arr[2*i+2];
		out_arr[idx].c[0] = c_arr[2*i+0];
		out_arr[idx].c[1] = c_arr[2*i+1];
		out_arr[idx].c[2] = c_arr[2*i+2];
		out_arr[idx].t[0] = t_arr[2*i+0];
		out_arr[idx].t[1] = t_arr[2*i+1];
		out_arr[idx].t[2] = t_arr[2*i+2];		
		
		idx++;

		out_arr[idx].v[0] = v_arr[2*i+1];
		out_arr[idx].v[1] = v_arr[2*i+2];
		out_arr[idx].v[2] = v_arr[2*i+3];
		out_arr[idx].n[0] = n_arr[2*i+1];
		out_arr[idx].n[1] = n_arr[2*i+2];
		out_arr[idx].n[2] = n_arr[2*i+3];
		out_arr[idx].c[0] = c_arr[2*i+1];
		out_arr[idx].c[1] = c_arr[2*i+2];
		out_arr[idx].c[2] = c_arr[2*i+3];
		out_arr[idx].t[0] = t_arr[2*i+1];
		out_arr[idx].t[1] = t_arr[2*i+2];
		out_arr[idx].t[2] = t_arr[2*i+3];		
		
		idx++;
	}
	
	return numTriangles;
}

