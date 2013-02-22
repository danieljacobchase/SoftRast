
/*//////////////////////////////////*/						
/* HELPER FUNCTION LIBRRARY: DEVICE */					
/*//////////////////////////////////*/	

/*//////////////////////////*/						
/* FUNCTION PREDECLARATIONS */					
/*//////////////////////////*/	

__device__ Vertex2D dev_newVertex2D( float x, float y );
__device__ Vertex3D dev_newVertex3D( float x, float y, float z );
__device__ Vertex4D dev_newVertex4D( float x, float y, float z, float w );
__device__ Color4D dev_newColor4D( float r, float g, float b, float a );
__device__ Matrix2D dev_newMatrix2D( Vertex2D a, Vertex2D b );
__device__ Matrix3D dev_newMatrix3D( Vertex3D a, Vertex3D b, Vertex3D c );
__device__ Matrix4D dev_newMatrix4D( Vertex4D a, Vertex4D b, Vertex4D c, Vertex4D d );

__device__ void dev_normalizeNormal2D( Vertex2D *n );
__device__ void dev_normalizeNormal3D( Vertex3D *n );
__device__ void dev_normalizeNormal4D( Vertex4D *n );
__device__ void dev_normalizeW4D( Vertex4D *n );
__device__ Vertex4D dev_normalizeW4D( Vertex4D n );
__device__ Vertex2D dev_normalizeNormal2D( Vertex2D n );
__device__ Vertex3D dev_normalizeNormal3D( Vertex3D n );
__device__ Vertex4D dev_normalizeNormal4D( Vertex4D n );
__device__ Matrix2D dev_normalizeMatrix2D( Matrix2D m );
__device__ Matrix3D dev_normalizeMatrix3D( Matrix3D m );
__device__ Matrix4D dev_normalizeMatrix4D( Matrix4D m );

__device__ Vertex2D dev_translateVertex2D( Vertex2D v, float s );
__device__ Vertex3D dev_translateVertex3D( Vertex3D v, float s );
__device__ Vertex4D dev_translateVertex4D( Vertex4D v, float s );
__device__ Vertex2D dev_scaleVertex2D( Vertex2D v, float s );
__device__ Vertex3D dev_scaleVertex3D( Vertex3D v, float s );
__device__ Vertex4D dev_scaleVertex4D( Vertex4D v, float s );

__device__ Matrix4D dev_scale(float x, float y, float z);
__device__ Matrix4D dev_translate(float x, float y, float z);
__device__ Matrix4D dev_rotate_x(float angle);
__device__ Matrix4D dev_rotate_y(float angle);
__device__ Matrix4D dev_rotate_z(float angle);

__device__ Matrix4D dev_loadIdentity4D();
__device__ Matrix3D dev_loadIdentity3D();
__device__ Matrix4D dev_view_frustum( float angle, float ar, float n, float f );
__device__ Matrix4D dev_projection_perspective( float r, float l, float b, float t, float n, float f );
__device__ Matrix4D dev_projection_orthographic( float r, float l, float b, float t, float n, float f );

__device__ float dev_determinantMatrix4D( Matrix4D M );
__device__ float dev_determinantMatrix3D( Matrix3D M );
__device__ float dev_determinantMatrix2D( Matrix2D M );

__device__ Matrix4D dev_transposeMatrix4D( Matrix4D dev_M );
__device__ Matrix3D dev_transposeMatrix3D( Matrix3D dev_M );
__device__ Matrix4D dev_adjointMatrix4D( Matrix4D dev_M );
__device__ Matrix3D dev_adjointMatrix3D( Matrix3D dev_M );
__device__ Matrix4D dev_inverseMatrix4D( Matrix4D dev_M );
__device__ Matrix3D dev_inverseMatrix3D( Matrix3D dev_M );
__device__ Matrix4D dev_scalarAddMatrix4D( Matrix4D dev_M, float s );
__device__ Matrix3D dev_scalarAddMatrix3D( Matrix3D dev_M, float s );
__device__ Matrix4D dev_scalarMulMatrix4D( Matrix4D dev_M, float m );
__device__ Matrix3D dev_scalarMulMatrix3D( Matrix3D dev_M, float m );

__device__ Matrix4D dev_mulMatrix4D( Matrix4D M1, Matrix4D M2 ); //M1*M2
__device__ Matrix3D dev_mulMatrix3D( Matrix3D M1, Matrix3D M2 ); //M1*M2
__device__ void dev_mulMatrix4D( Matrix4D M1, Matrix4D* M2 ); //puts result of M1*M2 in M2
__device__ void dev_mulMatrix3D( Matrix3D M1, Matrix3D* M2 ); //puts result of M1*M2 in M2
__device__ Vertex4D dev_mulMatrix4DVertex4D( Matrix4D M, Vertex4D V );
__device__ Vertex3D dev_mulMatrix3DVertex3D( Matrix3D M, Vertex3D V );
__device__ void dev_mulMatrix4DVertex3D( Matrix4D M, Vertex4D* V );
__device__ void dev_mulMatrix3DVertex3D( Matrix3D M, Vertex3D* V );

__device__ float dev_lengthVector( Vertex3D V1 ); 
__device__ float dev_lengthVector( Vertex4D V1 ); 

__device__ Vertex3D dev_crossProduct( Vertex3D V1, Vertex3D V2 ); // V1 x V2
__device__ Vertex3D dev_crossProduct( Vertex4D V1, Vertex4D V2 ); // V1 x V2
__device__ Vertex3D dev_subVertex3D( Vertex3D V1, Vertex3D V2 ); // V1 - V2
__device__ Vertex4D dev_subVertex4D( Vertex4D V1, Vertex4D V2 ); // V1 - V2
__device__ Vertex3D dev_addVertex3D( Vertex3D V1, Vertex3D V2 ); // V1 + V2
__device__ Vertex4D dev_addVertex4D( Vertex4D V1, Vertex4D V2 ); // V1 + V2

__device__ float dev_getTriangleArea( Triangle T );
__device__ unsigned short dev_squareSubPixel( unsigned short pixel, unsigned short subpixel );

__device__ Matrix4D dev_quickMatrix4D( float a, float b, float c, float d,
						float e, float f, float g, float h,
						float i, float j, float k, float l,
						float m, float n, float o, float p);
__device__ Matrix3D dev_quickMatrix3D( float a, float b, float c,
						float d, float e, float f,
						float g, float h, float i);
__device__ Matrix2D dev_quickMatrix2D( float a, float b,
						float c, float d);
						
__device__ void dev_bitFieldOn( unsigned short *bitField, int index );
__device__ void dev_bitFieldOff( unsigned short *bitField, int index );
__device__ int dev_bitFieldGet( unsigned short *bitField, int index );
						
__device__ void dev_snapCoord( float input, short *px, short *sub_px, int grid_bits );
__device__ float dev_unSnapCoord( short px, short sub_px, int grid_bits );

__device__ int dev_minVertY( Vertex4D *v1, Color4D *c1, Vertex3D *t1, Vertex4D *v2, Color4D *c2, Vertex3D *t2 );
__device__ int dev_triTypeSort( Vertex4D *v1, Vertex4D *v2, Vertex4D *v3, Color4D *c1, Color4D *c2, Color4D *c3, 
								Vertex3D *t1, Vertex3D *t2, Vertex3D *t3 );
__device__ void dev_integrateFragment( Color4D* f_buf_d, Float1D* d_buf_d, Color4D* t_buf_d, int x_coord, int y_coord, int screen_width, int tex_width,
				float curr_z, float curr_cr, float curr_cg, float curr_cb, float curr_ca, float curr_tx, float curr_ty, int DO_TEXTURE, int DO_DEPTH_TEST );

/*//////////////////////*/						
/* FUNCTION DEFINITIONS */					
/*//////////////////////*/	
						
__device__ Vertex2D dev_newVertex2D( float x, float y )
{
	Vertex2D output;
	output.x = x;
	output.y = y;
	return output;
}

__device__ Vertex3D dev_newVertex3D( float x, float y, float z )
{
	Vertex3D output;
	output.x = x;
	output.y = y;
	output.z = z;	
	return output;
}

__device__ Vertex4D dev_newVertex4D( float x, float y, float z, float w )
{
	Vertex4D output;
	output.x = x;
	output.y = y;
	output.z = z;	
	output.w = w;	
	return output;
}

__device__ Color4D dev_newColor4D( float r, float g, float b, float a )
{
	Color4D output;
	output.r = r;
	output.g = g;
	output.b = b;	
	output.a = a;	
	return output;
}

__device__ Matrix2D dev_newMatrix2D( Vertex2D a, Vertex2D b )
{
	Matrix2D output;
	output.a = a;
	output.b = b;
	return output;
}

__device__ Matrix3D dev_newMatrix3D( Vertex3D a, Vertex3D b, Vertex3D c )
{
	Matrix3D output;
	output.a = a;
	output.b = b;
	output.c = c;	
	return output;
}

__device__ Matrix4D dev_newMatrix4D( Vertex4D a, Vertex4D b, Vertex4D c, Vertex4D d )
{
	Matrix4D output;
	output.a = a;
	output.b = b;
	output.c = c;	
	output.d = d;	
	return output;
}

__device__ void dev_normalizeNormal2D( Vertex2D *n )
{
  float d = n->x*n->x + n->y*n->y;
  d = sqrt(d);
  if (d != 0.0) d = 1.0/d;
  n->x *= d;
  n->y *= d;
}

__device__ void dev_normalizeNormal3D( Vertex3D *n )
{
  float d = n->x*n->x + n->y*n->y + n->z*n->z;
  d = sqrt(d);
  if (d != 0.0) d = 1.0/d;
  n->x *= d;
  n->y *= d;
  n->z *= d;
}

__device__ void dev_normalizeNormal4D( Vertex4D *n )
{
  float d = n->x*n->x + n->y*n->y + n->z*n->z + n->w*n->w;
  d = sqrt(d);
  if (d != 0.0) d = 1.0/d;
  n->x *= d;
  n->y *= d;
  n->z *= d;
  n->w *= d;
}

__device__ Vertex2D dev_normalizeNormal2D( Vertex2D n )
{
  float d = n.x*n.x + n.y*n.y;
  d = sqrt(d);
  if (d != 0.0) d = 1.0/d;
  return dev_newVertex2D(n.x*d, n.y*d);
}

__device__ Vertex3D dev_normalizeNormal3D( Vertex3D n )
{
  float d = n.x*n.x + n.y*n.y + n.z*n.z;
  d = sqrt(d);
  if (d != 0.0) d = 1.0/d;
  return dev_newVertex3D(n.x*d, n.y*d, n.z*d);
}

__device__ Vertex4D dev_normalizeNormal4D( Vertex4D n )
{
  float d = n.x*n.x + n.y*n.y + n.z*n.z + n.w*n.w;
  d = sqrt(d);
  if (d != 0.0) d = 1.0/d;
  return dev_newVertex4D(n.x*d, n.y*d, n.z*d, n.w*d);
}

__device__ Vertex4D dev_normalizeW4D( Vertex4D n )
{
	float w_inv = 1/n.w;
	return dev_newVertex4D(n.x*w_inv, n.y*w_inv, n.z*w_inv, 1.0);
}

__device__ void dev_normalizeW4D( Vertex4D* n )
{
	float w_inv = 1/n->w;
	n->x = n->x*w_inv;
	n->y = n->y*w_inv;
	n->z = n->z*w_inv;
	n->w = 1.0;
}

__device__ Matrix2D dev_normalizeMatrix2D( Matrix2D m )
{
	return dev_newMatrix2D(
		dev_normalizeNormal2D( m.a ),
		dev_normalizeNormal2D( m.b )
	);
}

__device__ Matrix3D dev_normalizeMatrix3D( Matrix3D m )
{
	return dev_newMatrix3D(
		dev_normalizeNormal3D( m.a ),
		dev_normalizeNormal3D( m.b ),
		dev_normalizeNormal3D( m.c )
	);
}

__device__ Matrix4D dev_normalizeMatrix4D( Matrix4D m )
{
	return dev_newMatrix4D(
		dev_normalizeNormal4D( m.a ),
		dev_normalizeNormal4D( m.b ),
		dev_normalizeNormal4D( m.c ),
		dev_normalizeNormal4D( m.d )
	);
}

__device__ Vertex2D dev_translateVertex2D( Vertex2D v, float s ){ return dev_newVertex2D( v.x+s, v.y+s ); }
__device__ Vertex3D dev_translateVertex3D( Vertex3D v, float s ){ return dev_newVertex3D( v.x+s, v.y+s, v.z+s ); }
__device__ Vertex4D dev_translateVertex4D( Vertex4D v, float s ){ return dev_newVertex4D( v.x+s, v.y+s, v.z+s, v.w+s ); }
__device__ Vertex2D dev_scaleVertex2D( Vertex2D v, float s ){ return dev_newVertex2D( v.x*s, v.y*s ); }
__device__ Vertex3D dev_scaleVertex3D( Vertex3D v, float s ){ return dev_newVertex3D( v.x*s, v.y*s, v.z*s ); }
__device__ Vertex4D dev_scaleVertex4D( Vertex4D v, float s ){ return dev_newVertex4D( v.x*s, v.y*s, v.z*s, v.w*s ); }

/* angle is angle of view; ar is aspect ratio */
__device__ Matrix4D dev_view_frustum( float angle, float ar, float n, float f ) 
{
	//1 trig, 2 div, 6 mul, 3 add
	float div1 = 1/(f-n);
	float div2 = 1/tan(angle);
    return dev_newMatrix4D(
        dev_newVertex4D(1.0*div2,   0.0, 		0.0, 			0.0),
        dev_newVertex4D(0.0, 		ar*div2,  	0.0, 			0.0),
        dev_newVertex4D(0.0, 		0.0,    	(f+n)*div1, 	1.0),
        dev_newVertex4D(0.0, 		0.0, 		-2.0*f*n*div1, 	0.0)
    );
}

__device__ Matrix4D dev_projection_perspective( float r, float l, float b, float t, float n, float f )
{
	//3 div, 9 mul, 9 add
	float div1 = 1/(f-n);
	float div2 = 1/(t-b);
	float div3 = 1/(r-l);
    return dev_newMatrix4D(
        dev_newVertex4D(2*div3,  		0.0, 			0.0, 			0.0),
        dev_newVertex4D(0.0, 			2*n*div2,  		0.0, 			0.0),
        dev_newVertex4D(-(r+l)*div3, 	-(t+b)*div2, 	(f+n)*div1, 	1.0),
        dev_newVertex4D(0.0, 			0.0, 			-2.0*f*n*div1, 	0.0)
    );
}

__device__ Matrix4D dev_projection_orthographic( float r, float l, float b, float t, float n, float f )
{
	//3 div, 5 mult, 7 add
	float div1 = 1/(f-n);
	float div2 = 1/(t-b);
	float div3 = 1/(r-l);
    return dev_newMatrix4D(
        dev_newVertex4D(2*div3,  		0.0, 			0.0, 		0.0),
        dev_newVertex4D(0.0, 			2*div2,  		0.0, 		0.0),
        dev_newVertex4D(0.0, 			0.0, 			div1, 		0.0),
        dev_newVertex4D(-(r+l)*div3, 	-(t+b)*div2, 	-n*div1, 	1.0)
    );
}

__device__ Matrix4D dev_transposeMatrix4D( Matrix4D M )
{
    return dev_newMatrix4D(
        dev_newVertex4D(M.a.x, M.b.x, M.c.x, M.d.x),
        dev_newVertex4D(M.a.y, M.b.y, M.c.y, M.d.y),
        dev_newVertex4D(M.a.z, M.b.z, M.c.z, M.d.z),
        dev_newVertex4D(M.a.w, M.b.w, M.c.w, M.d.w)
    );
}

__device__ Matrix3D dev_transposeMatrix3D( Matrix3D M )
{
    return dev_newMatrix3D(
        dev_newVertex3D(M.a.x, M.b.x, M.c.x),
        dev_newVertex3D(M.a.y, M.b.y, M.c.y),
        dev_newVertex3D(M.a.z, M.b.z, M.c.z)
    );
}

__device__ Matrix4D dev_adjointMatrix4D( Matrix4D M )
{
	//144 mul, 54 add
	return dev_transposeMatrix4D( dev_newMatrix4D(
		dev_newVertex4D( 
			 dev_determinantMatrix3D( dev_quickMatrix3D( M.b.y, M.c.y, M.d.y, M.b.z, M.c.z, M.d.z, M.b.w, M.c.w, M.d.w ) ), 
			-dev_determinantMatrix3D( dev_quickMatrix3D( M.b.x, M.c.x, M.d.x, M.b.z, M.c.z, M.d.z, M.b.w, M.c.w, M.d.w ) ), 
			 dev_determinantMatrix3D( dev_quickMatrix3D( M.b.x, M.c.x, M.d.x, M.b.y, M.c.y, M.d.y, M.b.w, M.c.w, M.d.w ) ), 
			-dev_determinantMatrix3D( dev_quickMatrix3D( M.b.x, M.c.x, M.d.x, M.b.y, M.c.y, M.d.y, M.b.z, M.c.z, M.d.z ) )
		), 
		dev_newVertex4D(
			-dev_determinantMatrix3D( dev_quickMatrix3D( M.a.y, M.c.y, M.d.y, M.a.z, M.c.z, M.d.z, M.a.w, M.c.w, M.d.w ) ), 
			 dev_determinantMatrix3D( dev_quickMatrix3D( M.a.x, M.c.x, M.d.x, M.a.z, M.c.z, M.d.z, M.a.w, M.c.w, M.d.w ) ), 
			-dev_determinantMatrix3D( dev_quickMatrix3D( M.a.x, M.c.x, M.d.x, M.a.y, M.c.y, M.d.y, M.a.w, M.c.w, M.d.w ) ), 
			 dev_determinantMatrix3D( dev_quickMatrix3D( M.a.x, M.c.x, M.d.x, M.a.y, M.c.y, M.d.y, M.a.z, M.c.z, M.d.z ) )
		),
		dev_newVertex4D( 
			 dev_determinantMatrix3D( dev_quickMatrix3D( M.a.y, M.b.y, M.d.y, M.a.z, M.b.z, M.d.z, M.a.w, M.b.w, M.d.w ) ), 
			-dev_determinantMatrix3D( dev_quickMatrix3D( M.a.x, M.b.x, M.d.x, M.a.z, M.b.z, M.d.z, M.a.w, M.b.w, M.d.w ) ), 
			 dev_determinantMatrix3D( dev_quickMatrix3D( M.a.x, M.b.x, M.d.x, M.a.y, M.b.y, M.d.y, M.a.w, M.b.w, M.d.w ) ), 
			-dev_determinantMatrix3D( dev_quickMatrix3D( M.a.x, M.b.x, M.d.x, M.a.y, M.b.y, M.d.y, M.a.z, M.b.z, M.d.z ) )
		), 
		dev_newVertex4D(
			-dev_determinantMatrix3D( dev_quickMatrix3D( M.a.y, M.b.y, M.c.y, M.a.z, M.b.z, M.c.z, M.a.w, M.b.w, M.c.w ) ), 
			 dev_determinantMatrix3D( dev_quickMatrix3D( M.a.x, M.b.x, M.c.x, M.a.z, M.b.z, M.c.z, M.a.w, M.b.w, M.c.w ) ), 
			-dev_determinantMatrix3D( dev_quickMatrix3D( M.a.x, M.b.x, M.c.x, M.a.y, M.b.y, M.c.y, M.a.w, M.b.w, M.c.w ) ), 
			 dev_determinantMatrix3D( dev_quickMatrix3D( M.a.x, M.b.x, M.c.x, M.a.y, M.b.y, M.c.y, M.a.z, M.b.z, M.c.z ) )
		)
	) );
}

__device__ Matrix3D dev_adjointMatrix3D( Matrix3D M )
{
	//18 mul, 9 add
	return dev_transposeMatrix3D( dev_newMatrix3D(
		dev_newVertex3D( 
			 dev_determinantMatrix2D( dev_quickMatrix2D( M.b.y, M.c.y, M.b.z, M.c.z) ), 
			-dev_determinantMatrix2D( dev_quickMatrix2D( M.b.x, M.c.x, M.b.z, M.c.z) ), 
			 dev_determinantMatrix2D( dev_quickMatrix2D( M.b.x, M.c.x, M.b.y, M.c.y) ) 
		), 
		dev_newVertex3D(
			-dev_determinantMatrix2D( dev_quickMatrix2D( M.a.y, M.c.y, M.a.z, M.c.z ) ), 
			 dev_determinantMatrix2D( dev_quickMatrix2D( M.a.x, M.c.x, M.a.z, M.c.z ) ), 
			-dev_determinantMatrix2D( dev_quickMatrix2D( M.a.x, M.c.x, M.a.y, M.c.y ) ) 
		),
		dev_newVertex3D(
			 dev_determinantMatrix2D( dev_quickMatrix2D( M.a.y, M.b.y, M.a.z, M.b.z ) ), 
			-dev_determinantMatrix2D( dev_quickMatrix2D( M.a.x, M.b.x, M.a.z, M.b.z ) ), 
			 dev_determinantMatrix2D( dev_quickMatrix2D( M.a.x, M.b.x, M.a.y, M.b.y ) )
		)
	) );
}

__device__ Matrix3D dev_adjunctMatrix3D( Matrix3D M )
{
	//18 mul, 9 add
    return dev_newMatrix3D(
        dev_newVertex3D((M.b.y*M.c.z-M.c.y*M.b.z),(M.b.x*M.c.z-M.c.x*M.b.z),(M.b.x*M.c.y-M.c.x*M.b.y)),
		dev_newVertex3D((M.a.y*M.c.z-M.c.y*M.a.z),(M.a.x*M.c.z-M.c.x*M.a.z),(M.a.x*M.c.y-M.c.x*M.a.y)), 
		dev_newVertex3D((M.a.y*M.b.z-M.b.y*M.a.z),(M.a.x*M.b.z-M.b.x*M.a.z),(M.a.x*M.b.y-M.b.x*M.a.y))
	);					
}

__device__ Matrix4D dev_inverseMatrix4D( Matrix4D M )
{
	//1 div, 136 mul, 64 add
	float det, inv; 
	det = dev_determinantMatrix4D(M);
	if(det == 0.0) det += 0.000001;
	inv = 1/det;
	return dev_scalarMulMatrix4D( dev_adjointMatrix4D(M), inv );
}

__device__ Matrix3D dev_inverseMatrix3D( Matrix3D M )
{
	//1 div, 39 mul, 14 add
	float det, inv; 
	det = dev_determinantMatrix3D(M);
	if(det == 0.0) det += 0.000001;
	inv = 1/det;
	return dev_scalarMulMatrix3D( dev_adjointMatrix3D(M), inv );
}

__device__ float dev_determinantMatrix4D( Matrix4D M )
{
	//40 mult, 23 add
    return M.a.x*dev_determinantMatrix3D( dev_quickMatrix3D( M.b.y, M.c.y, M.d.y, M.b.z, M.c.z, M.d.z, M.b.w, M.c.w, M.d.w ) ) \
		  -M.b.x*dev_determinantMatrix3D( dev_quickMatrix3D( M.a.y, M.c.y, M.d.y, M.a.z, M.c.z, M.d.z, M.a.w, M.c.w, M.d.w ) ) \
		  +M.c.x*dev_determinantMatrix3D( dev_quickMatrix3D( M.a.y, M.b.y, M.d.y, M.a.z, M.b.z, M.d.z, M.a.w, M.b.w, M.d.w ) ) \
		  -M.d.x*dev_determinantMatrix3D( dev_quickMatrix3D( M.a.y, M.b.y, M.c.y, M.a.z, M.b.z, M.c.z, M.a.w, M.b.w, M.c.w ) );
}

__device__ float dev_determinantMatrix3D( Matrix3D M )
{
	//9 mult, 5 add
    return M.a.x*dev_determinantMatrix2D( dev_quickMatrix2D( M.b.y, M.c.y, M.b.z, M.c.z ) ) \
		  -M.b.x*dev_determinantMatrix2D( dev_quickMatrix2D( M.a.y, M.c.y, M.a.z, M.c.z ) ) \
		  +M.c.x*dev_determinantMatrix2D( dev_quickMatrix2D( M.a.y, M.b.y, M.a.z, M.b.z ) );
}

__device__ float dev_determinantMatrix2D( Matrix2D M )
{
	//2 mul, 1 add
	return M.a.x*M.b.y - M.b.x*M.a.y;
}

__device__ Matrix3D dev_scalarAddMatrix3D( Matrix3D M, float s )
{
	//9 add
	return dev_newMatrix3D(
		dev_newVertex3D(M.a.x+s, M.a.y+s, M.a.z+s),
		dev_newVertex3D(M.b.x+s, M.b.y+s, M.b.z+s),
		dev_newVertex3D(M.c.x+s, M.c.y+s, M.c.z+s)
	);
}

__device__ Matrix4D dev_scalarAddMatrix4D( Matrix4D M, float s )
{
	//16 add
	return dev_newMatrix4D(
		dev_newVertex4D(M.a.x+s, M.a.y+s, M.a.z+s, M.a.w+s),
		dev_newVertex4D(M.b.x+s, M.b.y+s, M.b.z+s, M.b.w+s),
		dev_newVertex4D(M.c.x+s, M.c.y+s, M.c.z+s, M.c.w+s),
		dev_newVertex4D(M.d.x+s, M.d.y+s, M.d.z+s, M.d.w+s)
	);
}

__device__ Matrix3D dev_scalarMulMatrix3D( Matrix3D M, float m )
{
	//9 mul
	return dev_newMatrix3D(
		dev_newVertex3D(M.a.x*m, M.a.y*m, M.a.z*m),
		dev_newVertex3D(M.b.x*m, M.b.y*m, M.b.z*m),
		dev_newVertex3D(M.c.x*m, M.c.y*m, M.c.z*m)
	);
}

__device__ Matrix4D dev_scalarMulMatrix4D( Matrix4D M, float m )
{
	//16 mul
	return dev_newMatrix4D(
		dev_newVertex4D(M.a.x*m, M.a.y*m, M.a.z*m, M.a.w*m),
		dev_newVertex4D(M.b.x*m, M.b.y*m, M.b.z*m, M.b.w*m),
		dev_newVertex4D(M.c.x*m, M.c.y*m, M.c.z*m, M.c.w*m),
		dev_newVertex4D(M.d.x*m, M.d.y*m, M.d.z*m, M.d.w*m)
	);
}

__device__ Matrix4D dev_loadIdentity4D() 
{
    return dev_newMatrix4D(
        dev_newVertex4D(1.0, 0.0, 0.0, 0.0),
        dev_newVertex4D(0.0, 1.0, 0.0, 0.0),
        dev_newVertex4D(0.0, 0.0, 1.0, 0.0),
        dev_newVertex4D(0.0, 0.0, 0.0, 1.0)
    );
}

__device__ Matrix3D dev_loadIdentity3D() 
{
    return dev_newMatrix3D(
        dev_newVertex3D(1.0, 0.0, 0.0),
        dev_newVertex3D(0.0, 1.0, 0.0),
        dev_newVertex3D(0.0, 0.0, 1.0)
    );
}

__device__ Matrix4D dev_scale(float x, float y, float z)
{
    return dev_newMatrix4D(
        dev_newVertex4D(x,   0.0, 0.0, 0.0),
        dev_newVertex4D(0.0, y,   0.0, 0.0),
        dev_newVertex4D(0.0, 0.0, z,   0.0),
        dev_newVertex4D(0.0, 0.0, 0.0, 1.0)
    );
}

__device__ Matrix4D dev_translate(float x, float y, float z)
{
    return dev_newMatrix4D(
        dev_newVertex4D(1.0, 0.0, 0.0, 0.0),
        dev_newVertex4D(0.0, 1.0, 0.0, 0.0),
        dev_newVertex4D(0.0, 0.0, 1.0, 0.0),
        dev_newVertex4D(x,   y,   z,   1.0)
    );
}

__device__ Matrix4D dev_rotate_x(float angle)
{
    return dev_newMatrix4D(
        dev_newVertex4D(1.0,         0.0,          0.0, 0.0),
        dev_newVertex4D(0.0,  cos(angle),  -sin(angle), 0.0),
        dev_newVertex4D(0.0,  sin(angle),   cos(angle), 0.0),
        dev_newVertex4D(0.0,         0.0,          0.0, 1.0)
    );
}

__device__ Matrix4D dev_rotate_y(float angle)
{
    return dev_newMatrix4D(
        dev_newVertex4D(cos(angle),  0.0,  -sin(angle), 0.0),
        dev_newVertex4D(0.0,  		 1.0,  0.0, 		0.0),
        dev_newVertex4D(sin(angle),  0.0,  cos(angle), 	0.0),
        dev_newVertex4D(0.0,         0.0,  0.0, 		1.0)
    );
}

__device__ Matrix4D dev_rotate_z(float angle)
{
    return dev_newMatrix4D(
        dev_newVertex4D(cos(angle),  sin(angle),  0.0, 0.0),
        dev_newVertex4D(-sin(angle), cos(angle),  0.0, 0.0),
        dev_newVertex4D(0.0, 		 0.0,  		  1.0, 0.0),
        dev_newVertex4D(0.0,         0.0,         0.0, 1.0)
    );
}

__device__ Matrix4D dev_mulMatrix4D( Matrix4D M1, Matrix4D M2 )
{
	//64 mults, 48 adds
	return dev_newMatrix4D(
        dev_newVertex4D(
			(M2.a.x*M1.a.x + M2.a.y*M1.b.x + M2.a.z*M1.c.x + M2.a.w*M1.d.x), 
			(M2.a.x*M1.a.y + M2.a.y*M1.b.y + M2.a.z*M1.c.y + M2.a.w*M1.d.y), 
			(M2.a.x*M1.a.z + M2.a.y*M1.b.z + M2.a.z*M1.c.z + M2.a.w*M1.d.z),
			(M2.a.x*M1.a.w + M2.a.y*M1.b.w + M2.a.z*M1.c.w + M2.a.w*M1.d.w)
		),
        dev_newVertex4D(
			(M2.b.x*M1.a.x + M2.b.y*M1.b.x + M2.b.z*M1.c.x + M2.b.w*M1.d.x), 
			(M2.b.x*M1.a.y + M2.b.y*M1.b.y + M2.b.z*M1.c.y + M2.b.w*M1.d.y), 
			(M2.b.x*M1.a.z + M2.b.y*M1.b.z + M2.b.z*M1.c.z + M2.b.w*M1.d.z),
			(M2.b.x*M1.a.w + M2.b.y*M1.b.w + M2.b.z*M1.c.w + M2.b.w*M1.d.w)
		),
        dev_newVertex4D(
			(M2.c.x*M1.a.x + M2.c.y*M1.b.x + M2.c.z*M1.c.x + M2.c.w*M1.d.x), 
			(M2.c.x*M1.a.y + M2.c.y*M1.b.y + M2.c.z*M1.c.y + M2.c.w*M1.d.y), 
			(M2.c.x*M1.a.z + M2.c.y*M1.b.z + M2.c.z*M1.c.z + M2.c.w*M1.d.z),
			(M2.c.x*M1.a.w + M2.c.y*M1.b.w + M2.c.z*M1.c.w + M2.c.w*M1.d.w)
		),
        dev_newVertex4D(
			(M2.d.x*M1.a.x + M2.d.y*M1.b.x + M2.d.z*M1.c.x + M2.d.w*M1.d.x), 
			(M2.d.x*M1.a.y + M2.d.y*M1.b.y + M2.d.z*M1.c.y + M2.d.w*M1.d.y), 
			(M2.d.x*M1.a.z + M2.d.y*M1.b.z + M2.d.z*M1.c.z + M2.d.w*M1.d.z),
			(M2.d.x*M1.a.w + M2.d.y*M1.b.w + M2.d.z*M1.c.w + M2.d.w*M1.d.w)
		)				
	);
}

__device__ Matrix3D dev_mulMatrix3D( Matrix3D M1, Matrix3D M2 )
{
	//27 mults, 24 adds
	return dev_newMatrix3D(
        dev_newVertex3D(
			(M2.a.x*M1.a.x + M2.a.y*M1.b.x + M2.a.z*M1.c.x), 
			(M2.a.x*M1.a.y + M2.a.y*M1.b.y + M2.a.z*M1.c.y), 
			(M2.a.x*M1.a.z + M2.a.y*M1.b.z + M2.a.z*M1.c.z)
		),
        dev_newVertex3D(
			(M2.b.x*M1.a.x + M2.b.y*M1.b.x + M2.b.z*M1.c.x), 
			(M2.b.x*M1.a.y + M2.b.y*M1.b.y + M2.b.z*M1.c.y), 
			(M2.b.x*M1.a.z + M2.b.y*M1.b.z + M2.b.z*M1.c.z)
		),
        dev_newVertex3D(
			(M2.c.x*M1.a.x + M2.c.y*M1.b.x + M2.c.z*M1.c.x), 
			(M2.c.x*M1.a.y + M2.c.y*M1.b.y + M2.c.z*M1.c.y), 
			(M2.c.x*M1.a.z + M2.c.y*M1.b.z + M2.c.z*M1.c.z)
		)
	);
}

__device__ void dev_mulMatrix4D( Matrix4D M1, Matrix4D* M2 )
{
	//64 mults, 48 adds
	float v1 = M2->a.x*M1.a.x + M2->a.y*M1.b.x + M2->a.z*M1.c.x + M2->a.w*M1.d.x;
	float v2 = M2->a.x*M1.a.y + M2->a.y*M1.b.y + M2->a.z*M1.c.y + M2->a.w*M1.d.y; 
	float v3 = M2->a.x*M1.a.z + M2->a.y*M1.b.z + M2->a.z*M1.c.z + M2->a.w*M1.d.z;
	float v4 = M2->a.x*M1.a.w + M2->a.y*M1.b.w + M2->a.z*M1.c.w + M2->a.w*M1.d.w;

	float v5 = M2->b.x*M1.a.x + M2->b.y*M1.b.x + M2->b.z*M1.c.x + M2->b.w*M1.d.x; 
	float v6 = M2->b.x*M1.a.y + M2->b.y*M1.b.y + M2->b.z*M1.c.y + M2->b.w*M1.d.y; 
	float v7 = M2->b.x*M1.a.z + M2->b.y*M1.b.z + M2->b.z*M1.c.z + M2->b.w*M1.d.z;
	float v8 = M2->b.x*M1.a.w + M2->b.y*M1.b.w + M2->b.z*M1.c.w + M2->b.w*M1.d.w;

	float v9 = M2->c.x*M1.a.x + M2->c.y*M1.b.x + M2->c.z*M1.c.x + M2->c.w*M1.d.x;
	float v10 = M2->c.x*M1.a.y + M2->c.y*M1.b.y + M2->c.z*M1.c.y + M2->c.w*M1.d.y;
	float v11 = M2->c.x*M1.a.z + M2->c.y*M1.b.z + M2->c.z*M1.c.z + M2->c.w*M1.d.z;
	float v12 = M2->c.x*M1.a.w + M2->c.y*M1.b.w + M2->c.z*M1.c.w + M2->c.w*M1.d.w;

	float v13 = M2->d.x*M1.a.x + M2->d.y*M1.b.x + M2->d.z*M1.c.x + M2->d.w*M1.d.x;
	float v14 = M2->d.x*M1.a.y + M2->d.y*M1.b.y + M2->d.z*M1.c.y + M2->d.w*M1.d.y;
	float v15 = M2->d.x*M1.a.z + M2->d.y*M1.b.z + M2->d.z*M1.c.z + M2->d.w*M1.d.z;
	float v16 = M2->d.x*M1.a.w + M2->d.y*M1.b.w + M2->d.z*M1.c.w + M2->d.w*M1.d.w;
			
	M2->a.x = v1;
	M2->a.y = v2;
	M2->a.z = v3;
	M2->a.w = v4;
	
	M2->b.x = v5;
	M2->b.y = v6;
	M2->b.z = v7;
	M2->b.w = v8;

	M2->c.x = v9;
	M2->c.y = v10;
	M2->c.z = v11;
	M2->c.w = v12;

	M2->d.x = v13;
	M2->d.y = v14;
	M2->d.z = v15;
	M2->d.w = v16;	
}

__device__ void dev_mulMatrix3D( Matrix3D M1, Matrix3D* M2 )
{
	//27 mults, 24 adds
	float v1 = M2->a.x*M1.a.x + M2->a.y*M1.b.x + M2->a.z*M1.c.x;
	float v2 = M2->a.x*M1.a.y + M2->a.y*M1.b.y + M2->a.z*M1.c.y;
	float v3 = M2->a.x*M1.a.z + M2->a.y*M1.b.z + M2->a.z*M1.c.z;

	float v4 = M2->b.x*M1.a.x + M2->b.y*M1.b.x + M2->b.z*M1.c.x; 
	float v5 = M2->b.x*M1.a.y + M2->b.y*M1.b.y + M2->b.z*M1.c.y; 
	float v6 = M2->b.x*M1.a.z + M2->b.y*M1.b.z + M2->b.z*M1.c.z;

	float v7 = M2->c.x*M1.a.x + M2->c.y*M1.b.x + M2->c.z*M1.c.x; 
	float v8 = M2->c.x*M1.a.y + M2->c.y*M1.b.y + M2->c.z*M1.c.y; 
	float v9 = M2->c.x*M1.a.z + M2->c.y*M1.b.z + M2->c.z*M1.c.z;

	M2->a.x = v1;
	M2->a.y = v2;
	M2->a.z = v3;
	
	M2->b.x = v4;
	M2->b.y = v5;
	M2->b.z = v6;

	M2->c.x = v7;
	M2->c.y = v8;
	M2->c.z = v9;	
}

__device__ Vertex4D dev_mulMatrix4DVertex4D( Matrix4D M, Vertex4D V )
{
	//16 mults; 12 adds
	return dev_newVertex4D( 
		(M.a.x*V.x + M.b.x*V.y + M.c.x*V.z + M.d.x*V.w), 
		(M.a.y*V.x + M.b.y*V.y + M.c.y*V.z + M.d.y*V.w),
		(M.a.z*V.x + M.b.z*V.y + M.c.z*V.z + M.d.z*V.w),
		(M.a.w*V.x + M.b.w*V.y + M.c.w*V.z + M.d.w*V.w)
	);
}

__device__ Vertex3D dev_mulMatrix3DVertex3D( Matrix3D M, Vertex3D V )
{
	//16 mults; 12 adds
	return dev_newVertex3D( 
		(M.a.x*V.x + M.b.x*V.y + M.c.x*V.z), 
		(M.a.y*V.x + M.b.y*V.y + M.c.y*V.z),
		(M.a.z*V.x + M.b.z*V.y + M.c.z*V.z)
	);
}

__device__ void dev_mulMatrix4DVertex4D( Matrix4D M, Vertex4D* V )
{
	//16 mults; 12 adds
	float v1 = M.a.x*V->x + M.b.x*V->y + M.c.x*V->z + M.d.x*V->w; 
	float v2 = M.a.y*V->x + M.b.y*V->y + M.c.y*V->z + M.d.y*V->w;
	float v3 = M.a.z*V->x + M.b.z*V->y + M.c.z*V->z + M.d.z*V->w;
	float v4 = M.a.w*V->x + M.b.w*V->y + M.c.w*V->z + M.d.w*V->w;
	V->x = v1;
	V->y = v2;
	V->z = v3;
	V->w = v4;
}

__device__ void dev_mulMatrix3DVertex3D( Matrix3D M, Vertex3D* V )
{
	//16 mults; 12 adds
	float v1 = M.a.x*V->x + M.b.x*V->y + M.c.x*V->z;
	float v2 = M.a.y*V->x + M.b.y*V->y + M.c.y*V->z;
	float v3 = M.a.z*V->x + M.b.z*V->y + M.c.z*V->z;
	V->x = v1;
	V->y = v2;
	V->z = v3;
}

__device__ float dev_lengthVector( Vertex3D V1 ) 
{
	return sqrt(V1.x*V1.x + V1.y*V1.y + V1.z*V1.z);
}

__device__ float dev_lengthVector( Vertex4D V1 ) // V1 x V2
{
	return sqrt(V1.x*V1.x + V1.y*V1.y + V1.z*V1.z);
}

__device__ Vertex3D dev_crossProduct( Vertex3D V1, Vertex3D V2 ) // V1 x V2
{
	return dev_newVertex3D( (V1.y*V2.z - V1.z*V2.y), (V1.z*V2.x - V1.x*V2.z), (V1.x*V2.y - V1.y*V2.x) );
}

__device__ Vertex3D dev_crossProduct( Vertex4D V1, Vertex4D V2 ) // V1 x V2
{
	return dev_newVertex3D( (V1.y*V2.z - V1.z*V2.y), (V1.z*V2.x - V1.x*V2.z), (V1.x*V2.y - V1.y*V2.x) );
}

__device__ Vertex3D dev_subVertex3D( Vertex3D V1, Vertex3D V2 ) // V1 - V2
{
	return dev_newVertex3D( (V1.x-V2.x), (V1.y-V2.y), (V1.z-V2.z) );
}

__device__ Vertex4D dev_subVertex4D( Vertex4D V1, Vertex4D V2 ) // V1 - V2
{
	return dev_newVertex4D( (V1.x-V2.x), (V1.y-V2.y), (V1.z-V2.z), (V1.w-V2.w) );
}

__device__ Vertex3D dev_addVertex3D( Vertex3D V1, Vertex3D V2 ) // V1 + V2
{
	return dev_newVertex3D( (V1.x+V2.x), (V1.y+V2.y), (V1.z+V2.z) );
}

__device__ Vertex4D dev_addVertex4D( Vertex4D V1, Vertex4D V2 ) // V1 + V2
{
	return dev_newVertex4D( (V1.x+V2.x), (V1.y+V2.y), (V1.z+V2.z), (V1.w+V2.w) );
}

__device__ float dev_getTriangleArea( Triangle T )
{
	//1 sqrt, 10 mul, 11 add
	Vertex3D cp = dev_crossProduct( dev_subVertex4D( T.v[2], T.v[0] ), dev_subVertex4D( T.v[2], T.v[1] ) );
	if(cp.z < 0.0) return 0.5*sqrt( cp.x*cp.x + cp.y*cp.y + cp.z*cp.z );
	else return -0.5*sqrt( cp.x*cp.x + cp.y*cp.y + cp.z*cp.z );
}

__device__ unsigned short dev_squareSubPixel( unsigned short pixel, unsigned short subpixel )
{
	//4 mul, 2 add
	unsigned short div = subpixel >> 13;
	return unsigned short(pixel*pixel + 2*pixel*div + div*div); 
}

__device__ Matrix4D dev_quickMatrix4D(  float a, float b, float c, float d,
										float e, float f, float g, float h,
										float i, float j, float k, float l,
										float m, float n, float o, float p)
{
	return dev_newMatrix4D(
		dev_newVertex4D(a, e, i, m),
		dev_newVertex4D(b, f, j, n),
		dev_newVertex4D(c, g, k, o),
		dev_newVertex4D(d, h, l, p)
	);
}

__device__ Matrix3D dev_quickMatrix3D(  float a, float b, float c,
										float d, float e, float f,
										float g, float h, float i)
{
	return dev_newMatrix3D(
		dev_newVertex3D(a, d, g),
		dev_newVertex3D(b, e, h),
		dev_newVertex3D(c, f, i)
	);
}

__device__ Matrix2D dev_quickMatrix2D(  float a, float b,
										float c, float d)
{
	return dev_newMatrix2D(
		dev_newVertex2D(a, c),
		dev_newVertex2D(b, d)
	);
}

__device__ void dev_bitFieldOn( unsigned short *bitField, int index )
{	
	int off = index >> 4; //int divide by 16
	bitField[off] = bitField[off] | (1UL << (index & 15UL));
}

__device__ void dev_bitFieldOff( unsigned short *bitField, int index )
{
	int off = index >> 4; //int divide by 16
	bitField[off] = bitField[off] & ~(1UL << (index & 15UL));
}

__device__ int dev_bitFieldGet( unsigned short *bitField, int index )
{
	return ((bitField[index >> 4] & (1UL << (index & 15UL))) != 0);
}

/* assumes that all vertices have been frustum culled and projected into 2048*2048 viewport (therefore all values are [0, 2048]) */
__device__ void dev_snapCoord( float input, short *px, short *sub_px, int grid_bits )
{
	input *= (1UL << grid_bits);
	int tempInt = (int)input;
	*px = tempInt >> grid_bits;
	*sub_px = tempInt & ((1UL << grid_bits)-1);
}

__device__ float dev_unSnapCoord( short px, short sub_px, int grid_bits )
{ 
	return float(px+(float(sub_px)/(1UL << grid_bits))); 
}

__device__ int dev_triTypeSort( Vertex4D *v1, Vertex4D *v2, Vertex4D *v3, Color4D *c1, Color4D *c2, Color4D *c3, 
								Vertex3D *t1, Vertex3D *t2, Vertex3D *t3 )
{
	int tri_case;
	Vertex4D v_t = dev_newVertex4D(0, 0, 0, 0);
	Vertex4D *vt;
	vt = &v_t;
	Color4D c_t = dev_newColor4D(0, 0, 0, 0);
	Color4D *ct;
	ct = &c_t;
	Vertex3D t_t = dev_newVertex3D(0, 0, 0);
	Vertex3D *tt;
	tt = &t_t;
	
	dev_minVertY( v1, c1, t1, v2, c2, t2 );
	tri_case = dev_minVertY( v1, c1, t1, v3, c3, t3 );

	if(tri_case == 1){ 
		*vt = *v2; *v2 = *v3; *v3 = *vt;
		*ct = *c2; *c2 = *c3; *c3 = *ct; 
		*tt = *t2; *t2 = *t3; *t3 = *tt; 		
		if(v2->x < v3->x){
			tri_case++; 
			if(v2->x < v1->x) tri_case--;
		}
		else{
			if(v2->x < v1->x) tri_case++;
		}	
	} 
	else{
		tri_case = 1; 
		dev_minVertY( v2, c2, t2, v3, c3, t3 ); 
		if(v2->x < v3->x){
			tri_case++; 
			if(v2->x < v1->x) tri_case--;
		}
		else{
			if(v2->x < v1->x) tri_case++;
		}	
	}
	if(v1->y == v2->y) tri_case = 4;
	else if(v2->y == v3->y) tri_case = 3; 
	
	return tri_case;
}

__device__ int dev_minVertY( Vertex4D *v1, Color4D *c1, Vertex3D *t1, Vertex4D *v2, Color4D *c2, Vertex3D *t2 )
{
	Vertex4D v_t = dev_newVertex4D(0, 0, 0, 0);
	Vertex4D *vt;
	vt = &v_t;
	Color4D c_t = dev_newColor4D(0, 0, 0, 0);
	Color4D *ct;
	ct = &c_t;
	Vertex3D t_t = dev_newVertex3D(0, 0, 0);
	Vertex3D *tt;
	tt = &t_t;
	
	if(v1->y > v2->y){
		*vt = *v2; *v2 = *v1; *v1 = *vt;
		*ct = *c2; *c2 = *c1; *c1 = *ct;
		*tt = *t2; *t2 = *t1; *t1 = *tt;
		return 1;
	}
	else if(v1->y < v2->y){ 
		return 2; }
	else{
		if(v1->x > v2->x){ 
			*vt = *v2; *v2 = *v1; *v1 = *vt; 
			*ct = *c2; *c2 = *c1; *c1 = *ct; 
			*tt = *t2; *t2 = *t1; *t1 = *tt; 
		}
		return 3;
	}
}

__device__ void dev_integrateFragment( Color4D* f_buf_d, Float1D* d_buf_d, Color4D* t_buf_d, int x_coord, int y_coord, int screen_width, 
										int tex_width, float curr_z, float curr_cr, float curr_cg, float curr_cb, float curr_ca, 
										float curr_tx, float curr_ty, int DO_TEXTURE, int DO_DEPTH_TEST )
{
	int idx = x_coord + y_coord*screen_width;
	
	// depth test; if current depth is greater than value in depth buffer (and therefore this fragment is farther away than curren closest), do nothing
	if((!DO_DEPTH_TEST) || (curr_z < d_buf_d[idx].f)){
			
		float conv_ca;
		
		// update z buffer value
		d_buf_d[idx].f = curr_z;
		
		// if texture is not on, we are concerned only with color blending
		if(!DO_TEXTURE){
			
			// opacity test; if totally opaque, do not have to do a blend operation
			if(curr_ca == 1.0){ // no blending; replacing colors		
				f_buf_d[idx].r = curr_cr;
				f_buf_d[idx].g = curr_cg;
				f_buf_d[idx].b = curr_cb;
				f_buf_d[idx].a = curr_ca;
			}
			else{ // blending; using "over" operation
				conv_ca = 1.0-curr_ca;
				f_buf_d[idx].r = curr_cr*curr_ca + f_buf_d[idx].r*f_buf_d[idx].a*conv_ca;
				f_buf_d[idx].g = curr_cg*curr_ca + f_buf_d[idx].g*f_buf_d[idx].a*conv_ca;
				f_buf_d[idx].b = curr_cb*curr_ca + f_buf_d[idx].b*f_buf_d[idx].a*conv_ca;
				f_buf_d[idx].a = curr_ca + f_buf_d[idx].a*conv_ca;
			}
		}
		else{
		
			int tx_idx = (int)curr_tx + tex_width*(int)curr_ty;
			if(tx_idx < (tex_width*tex_width) && tx_idx >= 0){
		
				// opacity test; if totally opaque, do not have to do a blend operation
				if(curr_ca == 1.0){ // no blending; replacing colors
					f_buf_d[idx].r = t_buf_d[tx_idx].r;
					f_buf_d[idx].g = t_buf_d[tx_idx].g;
					f_buf_d[idx].b = t_buf_d[tx_idx].b;
					f_buf_d[idx].a = t_buf_d[tx_idx].a;
				}
				else{ // blending; using "over" operation
					conv_ca = 1.0-t_buf_d[tx_idx].a;
					f_buf_d[idx].r = t_buf_d[tx_idx].r*t_buf_d[tx_idx].a + f_buf_d[idx].r*f_buf_d[idx].a*conv_ca;
					f_buf_d[idx].g = t_buf_d[tx_idx].g*t_buf_d[tx_idx].a + f_buf_d[idx].g*f_buf_d[idx].a*conv_ca;
					f_buf_d[idx].b = t_buf_d[tx_idx].b*t_buf_d[tx_idx].a + f_buf_d[idx].b*f_buf_d[idx].a*conv_ca;
					f_buf_d[idx].a = t_buf_d[tx_idx].a + f_buf_d[idx].a*conv_ca;
				}	
			}
		}
	}
}
