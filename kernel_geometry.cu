#include <cuda.h>
#include <cuda_runtime_api.h>

/*/////////////////////////////////////////////////////////////////////////*/						
/* PERFORMS "VERTEX SHADING" ACCORDING TO PRECALCUATED MODELVIEW TRANSFORM */					
/*/////////////////////////////////////////////////////////////////////////*/	

__global__ 
void kernel_geometry( int num_tris, Triangle* triangles_d, Matrix4D modelView4D_d, 
						Matrix4D projection4D_d, Matrix4D viewport4D_d, int DO_PERSPECTIVE_CORRECT, Int1D* draw_bit_d ) 
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx<num_tris){ 
		
		// modelview transform
		triangles_d[idx].v[0] = dev_mulMatrix4DVertex4D( modelView4D_d, triangles_d[idx].v[0] ); 
		triangles_d[idx].v[1] = dev_mulMatrix4DVertex4D( modelView4D_d, triangles_d[idx].v[1] ); 
		triangles_d[idx].v[2] = dev_mulMatrix4DVertex4D( modelView4D_d, triangles_d[idx].v[2] ); 

		// projection transform
		triangles_d[idx].v[0] = dev_mulMatrix4DVertex4D( projection4D_d, triangles_d[idx].v[0] ); 
		triangles_d[idx].v[1] = dev_mulMatrix4DVertex4D( projection4D_d, triangles_d[idx].v[1] ); 
		triangles_d[idx].v[2] = dev_mulMatrix4DVertex4D( projection4D_d, triangles_d[idx].v[2] ); 
		
		// if perspective correct is turned on, adjust u,v,w texture coordinate values 
		if( DO_PERSPECTIVE_CORRECT ){
			triangles_d[idx].t[0] = dev_scaleVertex3D( triangles_d[idx].t[0], 1/triangles_d[idx].v[0].w ); 
			triangles_d[idx].t[1] = dev_scaleVertex3D( triangles_d[idx].t[1], 1/triangles_d[idx].v[1].w );
			triangles_d[idx].t[2] = dev_scaleVertex3D( triangles_d[idx].t[2], 1/triangles_d[idx].v[2].w );
		}
		
		// normalize vertex coordinates
		triangles_d[idx].v[0] = dev_normalizeW4D( triangles_d[idx].v[0] ); 
		triangles_d[idx].v[1] = dev_normalizeW4D( triangles_d[idx].v[1] );
		triangles_d[idx].v[2] = dev_normalizeW4D( triangles_d[idx].v[2] );
		
		// viewport transform
		triangles_d[idx].v[0] = dev_mulMatrix4DVertex4D( viewport4D_d, triangles_d[idx].v[0] ); 
		triangles_d[idx].v[1] = dev_mulMatrix4DVertex4D( viewport4D_d, triangles_d[idx].v[1] );
		triangles_d[idx].v[2] = dev_mulMatrix4DVertex4D( viewport4D_d, triangles_d[idx].v[2] );		
			
		draw_bit_d[idx].i = 1;
	}	
}