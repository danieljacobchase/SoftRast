#include <cuda.h>
#include <cuda_runtime_api.h>

/*/////////////////////////////////////////////////////////////////////////*/						
/*//////////////// PERFORMS RASTERIZATION BY TRIANGLE TYPE ////////////////*/					
/*/////////////////////////////////////////////////////////////////////////*/	

__global__ 
void kernel_rasterizer( int num_triangles, Triangle* triangles_d, Color4D* frame_buffer_d, Float1D* depth_buffer_d, 
						Color4D* tex_buffer_d, int screen_width, int tex_width, int DO_TEXTURE, int PERSPECTIVE_CORRECT, 
						int DO_DEPTH_TEST, Int1D* draw_bit_d ) 
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	
	if ( idx < num_triangles )
	{ 
		Vertex4D *v1, *v2, *v3;
		Vertex3D *t1, *t2, *t3;
		Color4D *c1, *c2, *c3;
		
		float dxdyA, dxdyB, dxdyC, dzdyA, dzdyB, dzdyC;
		float dcrdyA, dcrdyB, dcrdyC, dcgdyA, dcgdyB, dcgdyC, dcbdyA, dcbdyB, dcbdyC, dcadyA, dcadyB, dcadyC;
		float dtudyA, dtudyB, dtudyC, dtvdyA, dtvdyB, dtvdyC, dtwdyA, dtwdyB, dtwdyC;
		
		float dxdyL, dxdyR, dzdyL, dzdyR;
		float dcrdyL, dcrdyR, dcgdyL, dcgdyR, dcbdyL, dcbdyR, dcadyL, dcadyR;
		float dtudyL, dtudyR, dtvdyL, dtvdyR, dtwdyL, dtwdyR;	
		
		float dzRast, dcrRast, dcgRast, dcbRast, dcaRast, dtuRast, dtvRast, dtwRast;

		float curr_vz, curr_vy, curr_cr, curr_cg, curr_cb, curr_ca, curr_tu, curr_tv, curr_tw;	
		float left_vz, left_cr, left_cg, left_cb, left_ca, left_tu, left_tv, left_tw;	
		float right_vz, right_cr, right_cg, right_cb, right_ca, right_tu, right_tv, right_tw;	
		float diff_vx, diff_vy, left_vx, right_vx, last_l_vx, last_r_vx, left_vx_snap, right_vx_snap, inv_span_vx;
		float last_l_vy, last_l_vz, last_l_cr, last_l_cg, last_l_cb, last_l_ca, last_l_tu, last_l_tv, last_l_tw;
		float last_r_vz, last_r_cr, last_r_cg, last_r_cb, last_r_ca, last_r_tu, last_r_tv, last_r_tw;
		float start_y1, start_y2, left_x, right_x;
		float adjusted_tu, adjusted_tv;		
		float inv_A, inv_B, inv_C, inv_curr_tw = 0.0;
		int tri_case, x, y, tracker = 0; //1 = left higher; 2 = right higher; 3 = top flat; 4 = bottom flat

		v1 = &triangles_d[idx].v[0];
		v2 = &triangles_d[idx].v[1];
		v3 = &triangles_d[idx].v[2];
		c1 = &triangles_d[idx].c[0];
		c2 = &triangles_d[idx].c[1];
		c3 = &triangles_d[idx].c[2];
		t1 = &triangles_d[idx].t[0];
		t2 = &triangles_d[idx].t[1];
		t3 = &triangles_d[idx].t[2];

		/* sort verts by y */
		tri_case = dev_triTypeSort( v1, v2, v3, c1, c2, c3, t1, t2, t3 );

		inv_A = 1.0 / ((float)v2->y - (float)v1->y);
		inv_B = 1.0 / ((float)v3->y - (float)v1->y);
		inv_C = 1.0 / ((float)v3->y - (float)v2->y);
		
		/* y slopes */
		dxdyA = ((float)v2->x - (float)v1->x) * inv_A; 
		dxdyB = ((float)v3->x - (float)v1->x) * inv_B;  
		dxdyC = ((float)v3->x - (float)v2->x) * inv_C; 

		/* depth slopes */
		dzdyA = ((float)v2->z - (float)v1->z) * inv_A; 
		dzdyB = ((float)v3->z - (float)v1->z) * inv_B;  
		dzdyC = ((float)v3->z - (float)v2->z) * inv_C; 
		
		/* color slopes red */
		dcrdyA = ((float)c2->r - (float)c1->r) * inv_A; 
		dcrdyB = ((float)c3->r - (float)c1->r) * inv_B;  
		dcrdyC = ((float)c3->r - (float)c2->r) * inv_C; 

		/* color slopes green */
		dcgdyA = ((float)c2->g - (float)c1->g) * inv_A; 
		dcgdyB = ((float)c3->g - (float)c1->g) * inv_B;  
		dcgdyC = ((float)c3->g - (float)c2->g) * inv_C; 

		/* color slopes blue */
		dcbdyA = ((float)c2->b - (float)c1->b) * inv_A; 
		dcbdyB = ((float)c3->b - (float)c1->b) * inv_B;  
		dcbdyC = ((float)c3->b - (float)c2->b) * inv_C; 

		/* color slopes alpha */
		dcadyA = ((float)c2->a - (float)c1->a) * inv_A; 
		dcadyB = ((float)c3->a - (float)c1->a) * inv_B;  
		dcadyC = ((float)c3->a - (float)c2->a) * inv_C; 	
		
		/* texture slopes u */	
		dtudyA = ((float)t2->x - (float)t1->x) * inv_A; 
		dtudyB = ((float)t3->x - (float)t1->x) * inv_B;  
		dtudyC = ((float)t3->x - (float)t2->x) * inv_C; 	

		/* texture slopes v */	
		dtvdyA = ((float)t2->y - (float)t1->y) * inv_A; 
		dtvdyB = ((float)t3->y - (float)t1->y) * inv_B;  
		dtvdyC = ((float)t3->y - (float)t2->y) * inv_C; 

		/* texture slopes w */	
		dtwdyA = ((float)t2->z - (float)t1->z) * inv_A; 
		dtwdyB = ((float)t3->z - (float)t1->z) * inv_B;  
		dtwdyC = ((float)t3->z - (float)t2->z) * inv_C; 	
		
		// setup for bottom half of triangle
		if(tri_case == 1 || tri_case == 4){
			dxdyL  = dxdyB;
			dzdyL  = dzdyB;
			dcrdyL = dcrdyB;
			dcgdyL = dcgdyB;
			dcbdyL = dcbdyB;
			dcadyL = dcadyB;
			dtudyL = dtudyB;
			dtvdyL = dtvdyB;
			dtwdyL = dtwdyB;
			
			dxdyR  = dxdyA;
			dzdyR  = dzdyA;
			dcrdyR = dcrdyA;
			dcgdyR = dcgdyA;
			dcbdyR = dcbdyA;
			dcadyR = dcadyA;
			dtudyR = dtudyA;
			dtvdyR = dtvdyA;
			dtwdyR = dtwdyA;			
		}
		else if(tri_case == 2 || tri_case == 3){
			dxdyL  = dxdyA;
			dzdyL  = dzdyA;
			dcrdyL = dcrdyA;
			dcgdyL = dcgdyA;
			dcbdyL = dcbdyA;
			dcadyL = dcadyA;
			dtudyL = dtudyA;
			dtvdyL = dtvdyA;
			dtwdyL = dtwdyA;
			
			dxdyR  = dxdyB;
			dzdyR  = dzdyB;
			dcrdyR = dcrdyB;
			dcgdyR = dcgdyB;
			dcbdyR = dcbdyB;
			dcadyR = dcadyB;
			dtudyR = dtudyB;
			dtvdyR = dtvdyB;	
			dtwdyR = dtwdyB;
		}

		last_l_vy = v1->y;
		last_l_vx = last_r_vx = v1->x;
		
		last_l_vz = last_r_vz = v1->z;
		last_l_cr = last_r_cr = c1->r;
		last_l_cg = last_r_cg = c1->g;
		last_l_cb = last_r_cb = c1->b;
		last_l_ca = last_r_ca = c1->a;
		last_l_tu = last_r_tu = t1->x;
		last_l_tv = last_r_tv = t1->y;	
		last_l_tw = last_r_tw = t1->z;	

		start_y1 = v1->y;
		start_y2 = v2->y;
		left_x = v1->x;
		
		curr_vy = floor(last_l_vy + 0.5) + 0.5;
		diff_vy = curr_vy - last_l_vy; 	
		
		// walking up raster lines in bottom half of trangle (B - A)
		for(y=((int)(floor(start_y1 + 0.5) + 0.5)); y<=((int)(ceil(start_y2 - 0.5) - 0.5)); y++)
		{
			left_vx = (curr_vy - start_y1)*dxdyL + left_x;
			left_vx_snap = floor(left_vx + 0.5) + 0.5;			
			right_vx = (curr_vy - start_y1)*dxdyR + left_x;
			right_vx_snap = ceil(right_vx - 0.5) - 0.5;				

			// shadow rule 2: right shadow draws pixel
			if((right_vx - right_vx_snap) == 1.0){ right_vx_snap += 1.0; }
			
			// if == 0.0, only one fragment to draw; if less than zero, none to draw (triangle is thin and passes between point samples
			if((right_vx_snap - left_vx_snap) >= 0.0)
			{
				inv_span_vx = 1.0 / (right_vx - left_vx);	
				diff_vx = left_vx_snap - left_vx;	
				
				// interpolant value at left edge
				left_vz = curr_vz = last_l_vz + diff_vy*dzdyL;
				left_cr = curr_cr = last_l_cr + diff_vy*dcrdyL;
				left_cg = curr_cg = last_l_cg + diff_vy*dcgdyL;
				left_cb = curr_cb = last_l_cb + diff_vy*dcbdyL;
				left_ca = curr_ca = last_l_ca + diff_vy*dcadyL;
				left_tu = curr_tu = last_l_tu + diff_vy*dtudyL;
				left_tv = curr_tv = last_l_tv + diff_vy*dtvdyL;
				left_tw = curr_tw = last_l_tw + diff_vy*dtwdyL;

				// interpolant value at right edge
				right_vz = last_r_vz + diff_vy*dzdyR;
				right_cr = last_r_cr + diff_vy*dcrdyR;
				right_cg = last_r_cg + diff_vy*dcgdyR;
				right_cb = last_r_cb + diff_vy*dcbdyR;
				right_ca = last_r_ca + diff_vy*dcadyR;
				right_tu = last_r_tu + diff_vy*dtudyR;
				right_tv = last_r_tv + diff_vy*dtvdyR;
				right_tw = last_r_tw + diff_vy*dtwdyR;

				// interpolant slope across raster line
				dzRast  = (right_vz - left_vz) * inv_span_vx;
				dcrRast = (right_cr - left_cr) * inv_span_vx;
				dcgRast = (right_cg - left_cg) * inv_span_vx;
				dcbRast = (right_cb - left_cb) * inv_span_vx;
				dcaRast = (right_ca - left_ca) * inv_span_vx;
				dtuRast = (right_tu - left_tu) * inv_span_vx;
				dtvRast = (right_tv - left_tv) * inv_span_vx;
				dtwRast = (right_tw - left_tw) * inv_span_vx;

				// walking across raster line
				for(x=(int)left_vx_snap; x<=(int)right_vx_snap; x++)
				{
					curr_vz = curr_vz + diff_vx*dzRast;
					curr_cr = curr_cr + diff_vx*dcrRast;
					curr_cg = curr_cg + diff_vx*dcgRast;
					curr_cb = curr_cb + diff_vx*dcbRast;
					curr_ca = curr_ca + diff_vx*dcaRast;
					curr_tu = curr_tu + diff_vx*dtuRast;
					curr_tv = curr_tv + diff_vx*dtvRast;
					curr_tw = curr_tw + diff_vx*dtwRast;
					
					if(PERSPECTIVE_CORRECT){
						inv_curr_tw = 1.0/curr_tw;
						adjusted_tu = curr_tu * inv_curr_tw;
						adjusted_tv = curr_tv * inv_curr_tw;
					}
					else{
						adjusted_tu = curr_tu;
						adjusted_tv = curr_tv;				
					}
					
					if((x >= 0) && (x < screen_width) && ((int)curr_vy >= 0) && ((int)curr_vy < screen_width)){
						dev_integrateFragment( frame_buffer_d, depth_buffer_d, tex_buffer_d, x, (int)curr_vy, screen_width, tex_width,
											curr_vz, curr_cr, curr_cg, curr_cb, curr_ca, adjusted_tu, adjusted_tv, DO_TEXTURE, DO_DEPTH_TEST );
					}

					diff_vx = 1.0; //after doing non-integer step for first value, can step across pixels
				}
				
				// stepping up raster line and updating baseline values
				last_l_vx = left_vx;				
				last_l_vz = left_vz;
				last_l_cr = left_cr;
				last_l_cg = left_cg;
				last_l_cb = left_cb;
				last_l_ca = left_ca;
				last_l_tu = left_tu;
				last_l_tv = left_tv;
				last_l_tw = left_tw;
				
				last_r_vx = right_vx;				
				last_r_vz = right_vz;
				last_r_cr = right_cr;
				last_r_cg = right_cg;
				last_r_cb = right_cb;
				last_r_ca = right_ca;
				last_r_tu = right_tu;
				last_r_tv = right_tv;	
				last_r_tw = right_tw;			
				
				tracker++;
			}
			last_l_vy = curr_vy;
			curr_vy += 1.0;
			diff_vy = 1.0;
		}
		
		// setup for top half of triangle
		if(tri_case == 1 || tri_case == 4){
			dxdyL  = dxdyB;
			dzdyL  = dzdyB;
			dcrdyL = dcrdyB;
			dcgdyL = dcgdyB;
			dcbdyL = dcbdyB;
			dcadyL = dcadyB;
			dtudyL = dtudyB;
			dtvdyL = dtvdyB;
			dtwdyL = dtwdyB;
			
			dxdyR  = dxdyC;
			dzdyR  = dzdyC;
			dcrdyR = dcrdyC;
			dcgdyR = dcgdyC;
			dcbdyR = dcbdyC;
			dcadyR = dcadyC;
			dtudyR = dtudyC;
			dtvdyR = dtvdyC;	
			dtwdyR = dtwdyC;

			start_y1 = v2->y;
			start_y2 = v3->y;
			left_x = last_l_vx;
			right_x = v2->x;			
		}
		else if(tri_case == 2){
			dxdyL  = dxdyC;
			dzdyL  = dzdyC;
			dcrdyL = dcrdyC;
			dcgdyL = dcgdyC;
			dcbdyL = dcbdyC;
			dcadyL = dcadyC;
			dtudyL = dtudyC;
			dtvdyL = dtvdyC;
			dtwdyL = dtwdyC;
			
			dxdyR  = dxdyB;
			dzdyR  = dzdyB;
			dcrdyR = dcrdyB;
			dcgdyR = dcgdyB;
			dcbdyR = dcbdyB;
			dcadyR = dcadyB;
			dtudyR = dtudyB;
			dtvdyR = dtvdyB;	
			dtwdyR = dtwdyB;

			start_y1 = v2->y;
			start_y2 = v3->y;
			left_x = v2->x;
			right_x = last_r_vx;		
		}
		else if(tri_case == 3){
			dxdyL  = dxdyC;
			dzdyL  = dzdyC;
			dcrdyL = dcrdyC;
			dcgdyL = dcgdyC;
			dcbdyL = dcbdyC;
			dcadyL = dcadyC;
			dtudyL = dtudyC;
			dtvdyL = dtvdyC;
			dtwdyL = dtwdyC;
			
			dxdyR  = dxdyB;
			dzdyR  = dzdyB;
			dcrdyR = dcrdyB;
			dcgdyR = dcgdyB;
			dcbdyR = dcbdyB;
			dcadyR = dcadyB;
			dtudyR = dtudyB;
			dtvdyR = dtvdyB;	
			dtwdyR = dtwdyB;

			start_y1 = v2->y;
			start_y2 = v3->y;
			left_x = v1->x;
			right_x = v2->x;		
		}
		
		if(tri_case == 4){
			last_l_vy = v1->y;
			
			last_l_vx = v1->x;
			last_r_vx = v2->x;
			last_l_vz = v1->z;
			last_r_vz = v2->z;
			
			last_l_cr = c1->r;
			last_r_cr = c2->r;
			last_l_cg = c1->g;
			last_r_cg = c2->g;
			last_l_cb = c1->b;
			last_r_cb = c2->b;
			last_l_ca = c1->a;
			last_r_ca = c2->a;			
			
			last_l_tu = t1->x;
			last_r_tu = t2->x;
			last_l_tv = t1->y;
			last_r_tv = t2->y;				
			last_l_tw = t1->z;
			last_r_tw = t2->z;
			
			start_y1 = v1->y;
			start_y2 = v3->y;	
			left_x = v1->x;
			right_x = v2->x;
		}

		curr_vy = floor(last_l_vy + 0.5) + 0.5;
		diff_vy = curr_vy - last_l_vy; 	
		
		// walking up raster lines in top half of trangle (B - C)
		for(y=((int)(floor(start_y1 + 0.5) + 0.5)); y<=((int)(ceil(start_y2 - 0.5) - 0.5)); y++)
		{
			left_vx = (curr_vy - start_y1)*dxdyL + left_x;
			left_vx_snap = floor(left_vx + 0.5) + 0.5;			
			right_vx = (curr_vy - start_y1)*dxdyR + right_x;
			right_vx_snap = ceil(right_vx - 0.5) - 0.5;	

			// shadow rule: right shadow draws pixel
			if((right_vx - right_vx_snap) == 1.0){ right_vx_snap += 1.0; }
			
			// if == 0.0, only one fragment to draw; if less than zero, none to draw (triangle is thin and passes between point samples
			if((right_vx_snap - left_vx_snap) >= 0.0)
			{
				inv_span_vx = 1.0 / (right_vx - left_vx);
				diff_vx = left_vx_snap - left_vx;
				
				// interpolant value at left edge
				left_vz = curr_vz = last_l_vz + diff_vy*dzdyL;
				left_cr = curr_cr = last_l_cr + diff_vy*dcrdyL;
				left_cg = curr_cg = last_l_cg + diff_vy*dcgdyL;
				left_cb = curr_cb = last_l_cb + diff_vy*dcbdyL;
				left_ca = curr_ca = last_l_ca + diff_vy*dcadyL;
				left_tu = curr_tu = last_l_tu + diff_vy*dtudyL;
				left_tv = curr_tv = last_l_tv + diff_vy*dtvdyL;
				left_tw = curr_tw = last_l_tw + diff_vy*dtwdyL;

				// interpolant value at right edge
				right_vz = last_r_vz + diff_vy*dzdyR;
				right_cr = last_r_cr + diff_vy*dcrdyR;
				right_cg = last_r_cg + diff_vy*dcgdyR;
				right_cb = last_r_cb + diff_vy*dcbdyR;
				right_ca = last_r_ca + diff_vy*dcadyR;
				right_tu = last_r_tu + diff_vy*dtudyR;
				right_tv = last_r_tv + diff_vy*dtvdyR;
				right_tw = last_r_tw + diff_vy*dtwdyR;
				
				// interpolant slope across raster line
				dzRast  = (right_vz - left_vz) * inv_span_vx;
				dcrRast = (right_cr - left_cr) * inv_span_vx;
				dcgRast = (right_cg - left_cg) * inv_span_vx;
				dcbRast = (right_cb - left_cb) * inv_span_vx;
				dcaRast = (right_ca - left_ca) * inv_span_vx;
				dtuRast = (right_tu - left_tu) * inv_span_vx;
				dtvRast = (right_tv - left_tv) * inv_span_vx;
				dtwRast = (right_tw - left_tw) * inv_span_vx;
				
				// walking across raster line
				for(x=(int)left_vx_snap; x<=(int)right_vx_snap; x++)
				{
					curr_vz = curr_vz + diff_vx*dzRast;
					curr_cr = curr_cr + diff_vx*dcrRast;
					curr_cg = curr_cg + diff_vx*dcgRast;
					curr_cb = curr_cb + diff_vx*dcbRast;
					curr_ca = curr_ca + diff_vx*dcaRast;
					curr_tu = curr_tu + diff_vx*dtuRast;
					curr_tv = curr_tv + diff_vx*dtvRast;
					curr_tw = curr_tw + diff_vx*dtwRast;
					
					if(PERSPECTIVE_CORRECT){
						inv_curr_tw = 1.0/curr_tw;
						adjusted_tu = curr_tu * inv_curr_tw;
						adjusted_tv = curr_tv * inv_curr_tw;
					}
					else{
						adjusted_tu = curr_tu;
						adjusted_tv = curr_tv;				
					}	
					
					if((x >= 0) && (x < screen_width) && ((int)curr_vy >= 0) && ((int)curr_vy < screen_width)){
						dev_integrateFragment( frame_buffer_d, depth_buffer_d, tex_buffer_d, x, (int)curr_vy, screen_width, tex_width,
											curr_vz, curr_cr, curr_cg, curr_cb, curr_ca, adjusted_tu, adjusted_tv, DO_TEXTURE, DO_DEPTH_TEST );
					}

					diff_vx = 1.0; //after doing non-integer step for first value, can step across pixels
				}
				
				// stepping up raster line and updating baseline values
				last_l_vx = left_vx;
				last_l_vz = left_vz;
				last_l_cr = left_cr;
				last_l_cg = left_cg;
				last_l_cb = left_cb;
				last_l_ca = left_ca;
				last_l_tu = left_tu;
				last_l_tv = left_tv;
				last_l_tw = left_tw;
				
				last_r_vx = right_vx;				
				last_r_vz = right_vz;
				last_r_cr = right_cr;
				last_r_cg = right_cg;
				last_r_cb = right_cb;
				last_r_ca = right_ca;
				last_r_tu = right_tu;
				last_r_tv = right_tv;
				last_r_tw = right_tw;
				
				tracker++;
			}
			last_l_vy = curr_vy;
			curr_vy += 1.0;
			diff_vy = 1.0;
		}
	
		if(tracker != 0) draw_bit_d[idx].i = 1;
	}
}