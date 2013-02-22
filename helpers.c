/*////////////////////////////////*/						
/* HELPER FUNCTION LIBRRARY: HOST */					
/*////////////////////////////////*/	

#define CONVERT			3.14159265358979323/180
#define	PI				3.14159265358979323
#define DEFAULT_PARALLEL				1
#define DEFAULT_RADIUS					0.5
#define DEFAULT_DIVISIONS				15
#define DEFAULT_DO_PERSPECTIVE_CORRECT 	1
#define DEFAULT_DO_TEXTURE				0
#define DEFAULT_DO_DEPTH_TEST			1
#define DEFAULT_DO_AREA_TEST			0
#define DEFAULT_DO_GEOM_TEST 			0

/*//////////////////////////*/						
/* FUNCTION PREDECLARATIONS */					
/*//////////////////////////*/	

Vertex2D newVertex2D( float x, float y );
Vertex3D newVertex3D( float x, float y, float z );
Vertex4D newVertex4D( float x, float y, float z, float w );
Color4D newColor4D( float r, float g, float b, float a );
Matrix2D newMatrix2D( Vertex2D a, Vertex2D b );
Matrix3D newMatrix3D( Vertex3D a, Vertex3D b, Vertex3D c );
Matrix4D newMatrix4D( Vertex4D a, Vertex4D b, Vertex4D c, Vertex4D d );

void normalizeNormal2D( Vertex2D *n );
void normalizeNormal3D( Vertex3D *n );
void normalizeNormal4D( Vertex4D *n );
void normalizeW4D( Vertex4D *n );
Vertex4D normalizeW4D( Vertex4D n );
Vertex2D normalizeNormal2D( Vertex2D n );
Vertex3D normalizeNormal3D( Vertex3D n );
Vertex4D normalizeNormal4D( Vertex4D n );
Matrix2D normalizeMatrix2D( Matrix2D m );
Matrix3D normalizeMatrix3D( Matrix3D m );
Matrix4D normalizeMatrix4D( Matrix4D m );

Vertex2D translateVertex2D( Vertex2D v, float s );
Vertex3D translateVertex3D( Vertex3D v, float s );
Vertex4D translateVertex4D( Vertex4D v, float s );
Vertex2D scaleVertex2D( Vertex2D v, float s );
Vertex3D scaleVertex3D( Vertex3D v, float s );
Vertex4D scaleVertex4D( Vertex4D v, float s );

Matrix4D scale(float x, float y, float z);
Matrix4D translate(float x, float y, float z);
Matrix4D rotate_x(float angle);
Matrix4D rotate_y(float angle);
Matrix4D rotate_z(float angle);

Matrix4D loadIdentity4D();
Matrix3D loadIdentity3D();
Matrix4D view_frustum( float angle, float ar, float n, float f );
Matrix4D projection_perspective( float r, float l, float b, float t, float n, float f );
Matrix4D projection_orthographic( float r, float l, float b, float t, float n, float f );

float determinantMatrix4D( Matrix4D M );
float determinantMatrix3D( Matrix3D M );
float determinantMatrix2D( Matrix2D M );

Matrix4D transposeMatrix4D( Matrix4D M );
Matrix3D transposeMatrix3D( Matrix3D M );
Matrix4D adjointMatrix4D( Matrix4D M );
Matrix3D adjointMatrix3D( Matrix3D M );
Matrix4D inverseMatrix4D( Matrix4D M );
Matrix3D inverseMatrix3D( Matrix3D M );
Matrix4D scalarAddMatrix4D( Matrix4D M, float s );
Matrix3D scalarAddMatrix3D( Matrix3D M, float s );
Matrix4D scalarMulMatrix4D( Matrix4D M, float m );
Matrix3D scalarMulMatrix3D( Matrix3D M, float m );

Matrix4D mulMatrix4D( Matrix4D M1, Matrix4D M2 ); //M1*M2
Matrix3D mulMatrix3D( Matrix3D M1, Matrix3D M2 ); //M1*M2
void mulMatrix4D( Matrix4D M1, Matrix4D* M2 ); //puts result of M1*M2 in M2
void mulMatrix3D( Matrix3D M1, Matrix3D* M2 ); //puts result of M1*M2 in M2
Vertex4D mulMatrix4DVertex4D( Matrix4D M, Vertex4D V );
Vertex3D mulMatrix3DVertex3D( Matrix3D M, Vertex3D V );
void mulMatrix4DVertex3D( Matrix4D M, Vertex4D* V );
void mulMatrix3DVertex3D( Matrix3D M, Vertex3D* V );

void writeOutputState( char* fileName, Triangle *allTris, int tCount );
void writePPMFile( const char* filename, Color4D *cArr, int width, int height, int colorDepth );
void readPPMFile( const char* filename, Color4D **cArr, int *width, int *height );
void flipFrame( Color4D *cArr, int width, int height );
void printAppConfig( AppConfig ap );
void printMatrix4D( Matrix4D M );
void printMatrix3D( Matrix3D M );
void printVertex4D( Vertex4D V );
void printVertex3D( Vertex3D V );

float lengthVector( Vertex3D V1 ); 
float lengthVector( Vertex4D V1 ); 
Vertex3D crossProduct( Vertex3D V1, Vertex3D V2 ); // V1 x V2
Vertex3D crossProduct( Vertex4D V1, Vertex4D V2 ); // V1 x V2
Vertex3D subVertex3D( Vertex3D V1, Vertex3D V2 ); // V1 - V2
Vertex4D subVertex4D( Vertex4D V1, Vertex4D V2 ); // V1 - V2
Vertex3D addVertex3D( Vertex3D V1, Vertex3D V2 ); // V1 + V2
Vertex4D addVertex4D( Vertex4D V1, Vertex4D V2 ); // V1 + V2

float getTriangleArea( Triangle T );
int squareSubPixel( int pixel, int subpixel );

Matrix4D quickMatrix4D( float a, float b, float c, float d,
						float e, float f, float g, float h,
						float i, float j, float k, float l,
						float m, float n, float o, float p);
Matrix3D quickMatrix3D( float a, float b, float c,
						float d, float e, float f,
						float g, float h, float i);
Matrix2D quickMatrix2D( float a, float b,
						float c, float d);

void bitFieldOn( unsigned short *bitField, int index );
void bitFieldOff( unsigned short *bitField, int index );
int bitFieldGet( unsigned short *bitField, int index );

void snapCoord( float input, short *px, short *sub_px, int grid_bits );
float unSnapCoord( short px, short sub_px, int grid_bits );

float getTriangleAreaTest( float a, float b, float c, float d, float e, float f, float g, float h, float i );

int minVertY( Vertex4D *v1, Color4D *c1, Vertex3D *t1, Vertex4D *v2, Color4D *c2, Vertex3D *t2 );
int triTypeSort( Vertex4D *v1, Vertex4D *v2, Vertex4D *v3, Color4D *c1, Color4D *c2, Color4D *c3, Vertex3D *t1, Vertex3D *t2, Vertex3D *t3 );

void fineRast( Triangle* tri_culled_d, Color4D* frame_buffer_d, Float1D* depth_buffer_d, Color4D* tex_buffer_d, 
				int screen_width, int tex_width, int DO_TEXTURE, int PERSPECTIVE_CORRECT, int DO_DEPTH_TEST );		
void integrateFragment( Color4D* f_buf_d, Float1D* d_buf_d, Color4D* t_buf_d, int x_coord, int y_coord, int screen_width, int tex_width,
						float curr_z, float curr_cr, float curr_cg, float curr_cb, float curr_ca, float curr_tx, float curr_ty, int DO_TEXTURE, int DO_DEPTH_TEST );
void performanceTesterRast( int num_triangles, Triangle* triangles_d, Color4D* frame_buffer_d, Float1D* depth_buffer_d, 
						Color4D* tex_buffer_d, int screen_width, int tex_width, int DO_TEXTURE, int PERSPECTIVE_CORRECT, 
						int DO_DEPTH_TEST, Int1D* draw_bit_d, char* name );
void buildCheckerBoard( int textureSize, int tileSize, Color4D *tex_buffer );

void parseArgs(int argc, char** argv, AppConfig &ap );				
AppConfig initialize(int argc, char** argv);

float dCos( float angle );	
float dSin( float angle );

					
/*//////////////////////*/						
/* FUNCTION DEFINITIONS */					
/*//////////////////////*/	

Vertex2D newVertex2D( float x, float y )
{
	Vertex2D output;
	output.x = x;
	output.y = y;
	return output;
}

Vertex3D newVertex3D( float x, float y, float z )
{
	Vertex3D output;
	output.x = x;
	output.y = y;
	output.z = z;	
	return output;
}

Vertex4D newVertex4D( float x, float y, float z, float w )
{
	Vertex4D output;
	output.x = x;
	output.y = y;
	output.z = z;	
	output.w = w;	
	return output;
}

Color4D newColor4D( float r, float g, float b, float a )
{
	Color4D output;
	output.r = r;
	output.g = g;
	output.b = b;	
	output.a = a;	
	return output;
}

Matrix2D newMatrix2D( Vertex2D a, Vertex2D b )
{
	Matrix2D output;
	output.a = a;
	output.b = b;
	return output;
}

Matrix3D newMatrix3D( Vertex3D a, Vertex3D b, Vertex3D c )
{
	Matrix3D output;
	output.a = a;
	output.b = b;
	output.c = c;	
	return output;
}

Matrix4D newMatrix4D( Vertex4D a, Vertex4D b, Vertex4D c, Vertex4D d )
{
	Matrix4D output;
	output.a = a;
	output.b = b;
	output.c = c;	
	output.d = d;	
	return output;
}

void normalizeNormal2D( Vertex2D *n )
{
	float d = n->x*n->x + n->y*n->y;
	d = sqrt(d);
	if (d != 0.0) d = 1.0/d;
	n->x *= d;
	n->y *= d;
}

void normalizeNormal3D( Vertex3D *n )
{
	float d = n->x*n->x + n->y*n->y + n->z*n->z;
	d = sqrt(d);
	if (d != 0.0) d = 1.0/d;
	n->x *= d;
	n->y *= d;
	n->z *= d;
}

void normalizeNormal4D( Vertex4D *n )
{
	float d = n->x*n->x + n->y*n->y + n->z*n->z + n->w*n->w;
	d = sqrt(d);
	if (d != 0.0) d = 1.0/d;
	n->x *= d;
	n->y *= d;
	n->z *= d;
	n->w *= d;
}

Vertex2D normalizeNormal2D( Vertex2D n )
{
	float d = n.x*n.x + n.y*n.y;
	d = sqrt(d);
	if (d != 0.0) d = 1.0/d;
	return newVertex2D(n.x*d, n.y*d);
}

Vertex3D normalizeNormal3D( Vertex3D n )
{
	float d = n.x*n.x + n.y*n.y + n.z*n.z;
	d = sqrt(d);
	if (d != 0.0) d = 1.0/d;
	return newVertex3D(n.x*d, n.y*d, n.z*d);
}

Vertex4D normalizeNormal4D( Vertex4D n )
{
	float d = n.x*n.x + n.y*n.y + n.z*n.z + n.w*n.w;
	d = sqrt(d);
	if (d != 0.0) d = 1.0/d;
	return newVertex4D(n.x*d, n.y*d, n.z*d, n.w*d);
}

Vertex4D normalizeW4D( Vertex4D n )
{
	float w_inv = 1/n.w;
	return newVertex4D(n.x*w_inv, n.y*w_inv, n.z*w_inv, 1.0);
}

void normalizeW4D( Vertex4D* n )
{
	float w_inv = 1/n->w;
	n->x = n->x*w_inv;
	n->y = n->y*w_inv;
	n->z = n->z*w_inv;
	n->w = 1.0;
}

Matrix2D normalizeMatrix2D( Matrix2D m )
{
	return newMatrix2D(
		normalizeNormal2D( m.a ),
		normalizeNormal2D( m.b )
	);
}

Matrix3D normalizeMatrix3D( Matrix3D m )
{
	return newMatrix3D(
		normalizeNormal3D( m.a ),
		normalizeNormal3D( m.b ),
		normalizeNormal3D( m.c )
	);
}

Matrix4D normalizeMatrix4D( Matrix4D m )
{
	return newMatrix4D(
		normalizeNormal4D( m.a ),
		normalizeNormal4D( m.b ),
		normalizeNormal4D( m.c ),
		normalizeNormal4D( m.d )
	);
}

Vertex2D translateVertex2D( Vertex2D v, float s ){ return newVertex2D( v.x+s, v.y+s ); }
Vertex3D translateVertex3D( Vertex3D v, float s ){ return newVertex3D( v.x+s, v.y+s, v.z+s ); }
Vertex4D translateVertex4D( Vertex4D v, float s ){ return newVertex4D( v.x+s, v.y+s, v.z+s, v.w+s ); }
Vertex2D scaleVertex2D( Vertex2D v, float s ){ return newVertex2D( v.x*s, v.y*s ); }
Vertex3D scaleVertex3D( Vertex3D v, float s ){ return newVertex3D( v.x*s, v.y*s, v.z*s ); }
Vertex4D scaleVertex4D( Vertex4D v, float s ){ return newVertex4D( v.x*s, v.y*s, v.z*s, v.w*s ); }

/* angle is angle of view; ar is aspect ratio */
Matrix4D view_frustum( float angle, float ar, float n, float f ) 
{
	//1 trig, 2 div, 6 mul, 3 add
	float div1 = 1/(f-n);
	float div2 = 1/tan(angle);
    return newMatrix4D(
        newVertex4D(1.0*div2,   0.0, 		0.0, 			0.0),
        newVertex4D(0.0, 		ar*div2,  	0.0, 			0.0),
        newVertex4D(0.0, 		0.0,    	(f+n)*div1, 	1.0),
        newVertex4D(0.0, 		0.0, 		-2.0*f*n*div1, 	0.0)
    );
}

Matrix4D projection_perspective( float r, float l, float b, float t, float n, float f )
{
	//3 div, 9 mul, 9 add
	float div1 = 1/(f-n);
	float div2 = 1/(t-b);
	float div3 = 1/(r-l);
    return newMatrix4D(
        newVertex4D(2*n*div3,  		0.0, 			0.0, 			0.0),
        newVertex4D(0.0, 			2*n*div2,  		0.0, 			0.0),
        newVertex4D(-(r+l)*div3, 	-(t+b)*div2, 	(f+n)*div1, 	1.0),
        newVertex4D(0.0, 			0.0, 			-2.0*f*n*div1, 	0.0)
    );
}

Matrix4D projection_orthographic( float r, float l, float b, float t, float n, float f )
{
	//3 div, 5 mult, 7 add
	float div1 = 1/(f-n);
	float div2 = 1/(t-b);
	float div3 = 1/(r-l);
    return newMatrix4D(
        newVertex4D(2*div3,  		0.0, 			0.0, 			0.0),
        newVertex4D(0.0, 			2*div2,  		0.0, 			0.0),
        newVertex4D(0.0, 			0.0, 			-2*div1, 		0.0),
        newVertex4D(-(r+l)*div3, 	-(t+b)*div2, 	-(f+n)*div1, 	1.0)
    );
}

Matrix4D transposeMatrix4D( Matrix4D M )
{
    return newMatrix4D(
        newVertex4D(M.a.x, M.b.x, M.c.x, M.d.x),
        newVertex4D(M.a.y, M.b.y, M.c.y, M.d.y),
        newVertex4D(M.a.z, M.b.z, M.c.z, M.d.z),
        newVertex4D(M.a.w, M.b.w, M.c.w, M.d.w)
    );
}

Matrix3D transposeMatrix3D( Matrix3D M )
{
    return newMatrix3D(
        newVertex3D(M.a.x, M.b.x, M.c.x),
        newVertex3D(M.a.y, M.b.y, M.c.y),
        newVertex3D(M.a.z, M.b.z, M.c.z)
    );
}

Matrix4D adjointMatrix4D( Matrix4D M )
{
	//144 mul, 54 add
	return transposeMatrix4D( newMatrix4D(
		newVertex4D( 
			 determinantMatrix3D( quickMatrix3D( M.b.y, M.c.y, M.d.y, M.b.z, M.c.z, M.d.z, M.b.w, M.c.w, M.d.w ) ), 
			-determinantMatrix3D( quickMatrix3D( M.b.x, M.c.x, M.d.x, M.b.z, M.c.z, M.d.z, M.b.w, M.c.w, M.d.w ) ), 
			 determinantMatrix3D( quickMatrix3D( M.b.x, M.c.x, M.d.x, M.b.y, M.c.y, M.d.y, M.b.w, M.c.w, M.d.w ) ), 
			-determinantMatrix3D( quickMatrix3D( M.b.x, M.c.x, M.d.x, M.b.y, M.c.y, M.d.y, M.b.z, M.c.z, M.d.z ) )
		), 
		newVertex4D(
			-determinantMatrix3D( quickMatrix3D( M.a.y, M.c.y, M.d.y, M.a.z, M.c.z, M.d.z, M.a.w, M.c.w, M.d.w ) ), 
			 determinantMatrix3D( quickMatrix3D( M.a.x, M.c.x, M.d.x, M.a.z, M.c.z, M.d.z, M.a.w, M.c.w, M.d.w ) ), 
			-determinantMatrix3D( quickMatrix3D( M.a.x, M.c.x, M.d.x, M.a.y, M.c.y, M.d.y, M.a.w, M.c.w, M.d.w ) ), 
			 determinantMatrix3D( quickMatrix3D( M.a.x, M.c.x, M.d.x, M.a.y, M.c.y, M.d.y, M.a.z, M.c.z, M.d.z ) )
		),
		newVertex4D( 
			determinantMatrix3D( quickMatrix3D( M.a.y, M.b.y, M.d.y, M.a.z, M.b.z, M.d.z, M.a.w, M.b.w, M.d.w ) ), 
			-determinantMatrix3D( quickMatrix3D( M.a.x, M.b.x, M.d.x, M.a.z, M.b.z, M.d.z, M.a.w, M.b.w, M.d.w ) ), 
			 determinantMatrix3D( quickMatrix3D( M.a.x, M.b.x, M.d.x, M.a.y, M.b.y, M.d.y, M.a.w, M.b.w, M.d.w ) ), 
			-determinantMatrix3D( quickMatrix3D( M.a.x, M.b.x, M.d.x, M.a.y, M.b.y, M.d.y, M.a.z, M.b.z, M.d.z ) )
		), 
		newVertex4D(
			-determinantMatrix3D( quickMatrix3D( M.a.y, M.b.y, M.c.y, M.a.z, M.b.z, M.c.z, M.a.w, M.b.w, M.c.w ) ), 
			 determinantMatrix3D( quickMatrix3D( M.a.x, M.b.x, M.c.x, M.a.z, M.b.z, M.c.z, M.a.w, M.b.w, M.c.w ) ), 
			-determinantMatrix3D( quickMatrix3D( M.a.x, M.b.x, M.c.x, M.a.y, M.b.y, M.c.y, M.a.w, M.b.w, M.c.w ) ), 
			 determinantMatrix3D( quickMatrix3D( M.a.x, M.b.x, M.c.x, M.a.y, M.b.y, M.c.y, M.a.z, M.b.z, M.c.z ) )
		)
	) );
}

Matrix3D adjointMatrix3D( Matrix3D M )
{
	//18 mul, 9 add
	return transposeMatrix3D( newMatrix3D(
		newVertex3D( 
			 determinantMatrix2D( quickMatrix2D( M.b.y, M.c.y, M.b.z, M.c.z) ), 
			-determinantMatrix2D( quickMatrix2D( M.b.x, M.c.x, M.b.z, M.c.z) ), 
			 determinantMatrix2D( quickMatrix2D( M.b.x, M.c.x, M.b.y, M.c.y) ) 
		), 
		newVertex3D(
			-determinantMatrix2D( quickMatrix2D( M.a.y, M.c.y, M.a.z, M.c.z ) ), 
			 determinantMatrix2D( quickMatrix2D( M.a.x, M.c.x, M.a.z, M.c.z ) ), 
			-determinantMatrix2D( quickMatrix2D( M.a.x, M.c.x, M.a.y, M.c.y ) ) 
		),
		newVertex3D(
			 determinantMatrix2D( quickMatrix2D( M.a.y, M.b.y, M.a.z, M.b.z ) ), 
			-determinantMatrix2D( quickMatrix2D( M.a.x, M.b.x, M.a.z, M.b.z ) ), 
			 determinantMatrix2D( quickMatrix2D( M.a.x, M.b.x, M.a.y, M.b.y ) )
		)
	) );
}

Matrix3D adjunctMatrix3D( Matrix3D M )
{
	//18 mul, 9 add
    return newMatrix3D(
        newVertex3D((M.b.y*M.c.z-M.c.y*M.b.z),(M.b.x*M.c.z-M.c.x*M.b.z),(M.b.x*M.c.y-M.c.x*M.b.y)),
		newVertex3D((M.a.y*M.c.z-M.c.y*M.a.z),(M.a.x*M.c.z-M.c.x*M.a.z),(M.a.x*M.c.y-M.c.x*M.a.y)), 
		newVertex3D((M.a.y*M.b.z-M.b.y*M.a.z),(M.a.x*M.b.z-M.b.x*M.a.z),(M.a.x*M.b.y-M.b.x*M.a.y))
	);					
}

Matrix4D inverseMatrix4D( Matrix4D M )
{
	//1 div, 136 mul, 64 add
	float det, inv; 
	det = determinantMatrix4D(M);
	if(det == 0.0) det += 0.000001;
	inv = 1/det;
	return scalarMulMatrix4D( adjointMatrix4D(M), inv );
}

Matrix3D inverseMatrix3D( Matrix3D M )
{
	//1 div, 39 mul, 14 add
	float det, inv; 
	det = determinantMatrix3D(M);
	if(det == 0.0) det += 0.000001;
	inv = 1/det;
	return scalarMulMatrix3D( adjointMatrix3D(M), inv );
}

float determinantMatrix4D( Matrix4D M )
{
	//40 mult, 23 add
    return M.a.x*determinantMatrix3D( quickMatrix3D( M.b.y, M.c.y, M.d.y, M.b.z, M.c.z, M.d.z, M.b.w, M.c.w, M.d.w ) ) \
		  -M.b.x*determinantMatrix3D( quickMatrix3D( M.a.y, M.c.y, M.d.y, M.a.z, M.c.z, M.d.z, M.a.w, M.c.w, M.d.w ) ) \
		  +M.c.x*determinantMatrix3D( quickMatrix3D( M.a.y, M.b.y, M.d.y, M.a.z, M.b.z, M.d.z, M.a.w, M.b.w, M.d.w ) ) \
		  -M.d.x*determinantMatrix3D( quickMatrix3D( M.a.y, M.b.y, M.c.y, M.a.z, M.b.z, M.c.z, M.a.w, M.b.w, M.c.w ) );
}

float determinantMatrix3D( Matrix3D M )
{
	//9 mult, 5 add
    return M.a.x*determinantMatrix2D( quickMatrix2D( M.b.y, M.c.y, M.b.z, M.c.z ) ) \
		  -M.b.x*determinantMatrix2D( quickMatrix2D( M.a.y, M.c.y, M.a.z, M.c.z ) ) \
		  +M.c.x*determinantMatrix2D( quickMatrix2D( M.a.y, M.b.y, M.a.z, M.b.z ) );
}

float determinantMatrix2D( Matrix2D M )
{
	//2 mul, 1 add
	return M.a.x*M.b.y - M.b.x*M.a.y;
}

Matrix3D scalarAddMatrix3D( Matrix3D M, float s )
{
	//9 add
	return newMatrix3D(
		newVertex3D(M.a.x+s, M.a.y+s, M.a.z+s),
		newVertex3D(M.b.x+s, M.b.y+s, M.b.z+s),
		newVertex3D(M.c.x+s, M.c.y+s, M.c.z+s)
	);
}

Matrix4D scalarAddMatrix4D( Matrix4D M, float s )
{
	//16 add
	return newMatrix4D(
		newVertex4D(M.a.x+s, M.a.y+s, M.a.z+s, M.a.w+s),
		newVertex4D(M.b.x+s, M.b.y+s, M.b.z+s, M.b.w+s),
		newVertex4D(M.c.x+s, M.c.y+s, M.c.z+s, M.c.w+s),
		newVertex4D(M.d.x+s, M.d.y+s, M.d.z+s, M.d.w+s)
	);
}

Matrix3D scalarMulMatrix3D( Matrix3D M, float m )
{
	//9 mul
	return newMatrix3D(
		newVertex3D(M.a.x*m, M.a.y*m, M.a.z*m),
		newVertex3D(M.b.x*m, M.b.y*m, M.b.z*m),
		newVertex3D(M.c.x*m, M.c.y*m, M.c.z*m)
	);
}

Matrix4D scalarMulMatrix4D( Matrix4D M, float m )
{
	//16 mul
	return newMatrix4D(
		newVertex4D(M.a.x*m, M.a.y*m, M.a.z*m, M.a.w*m),
		newVertex4D(M.b.x*m, M.b.y*m, M.b.z*m, M.b.w*m),
		newVertex4D(M.c.x*m, M.c.y*m, M.c.z*m, M.c.w*m),
		newVertex4D(M.d.x*m, M.d.y*m, M.d.z*m, M.d.w*m)
	);
}

Matrix4D loadIdentity4D() 
{
    return newMatrix4D(
        newVertex4D(1.0, 0.0, 0.0, 0.0),
        newVertex4D(0.0, 1.0, 0.0, 0.0),
        newVertex4D(0.0, 0.0, 1.0, 0.0),
        newVertex4D(0.0, 0.0, 0.0, 1.0)
    );
}

Matrix3D loadIdentity3D() 
{
    return newMatrix3D(
        newVertex3D(1.0, 0.0, 0.0),
        newVertex3D(0.0, 1.0, 0.0),
        newVertex3D(0.0, 0.0, 1.0)
    );
}

Matrix4D scale(float x, float y, float z)
{
    return newMatrix4D(
        newVertex4D(x,   0.0, 0.0, 0.0),
        newVertex4D(0.0, y,   0.0, 0.0),
        newVertex4D(0.0, 0.0, z,   0.0),
        newVertex4D(0.0, 0.0, 0.0, 1.0)
    );
}

Matrix4D translate(float x, float y, float z)
{
    return newMatrix4D(
        newVertex4D(1.0, 0.0, 0.0, 0.0),
        newVertex4D(0.0, 1.0, 0.0, 0.0),
        newVertex4D(0.0, 0.0, 1.0, 0.0),
        newVertex4D(x,   y,   z,   1.0)
    );
}

Matrix4D rotate_x(float angle)
{
    return newMatrix4D(
        newVertex4D(1.0,         0.0,          0.0, 0.0),
        newVertex4D(0.0,  cos(angle),  -sin(angle), 0.0),
        newVertex4D(0.0,  sin(angle),   cos(angle), 0.0),
        newVertex4D(0.0,         0.0,          0.0, 1.0)
    );
}

Matrix4D rotate_y(float angle)
{
    return newMatrix4D(
        newVertex4D(cos(angle),  0.0,  -sin(angle), 0.0),
        newVertex4D(0.0,  		 1.0,  0.0, 		0.0),
        newVertex4D(sin(angle),  0.0,  cos(angle), 	0.0),
        newVertex4D(0.0,         0.0,  0.0, 		1.0)
    );
}

Matrix4D rotate_z(float angle)
{
    return newMatrix4D(
        newVertex4D(cos(angle),  sin(angle),  0.0, 0.0),
        newVertex4D(-sin(angle), cos(angle),  0.0, 0.0),
        newVertex4D(0.0, 		 0.0,  		  1.0, 0.0),
        newVertex4D(0.0,         0.0,         0.0, 1.0)
    );
}

Matrix4D mulMatrix4D( Matrix4D M1, Matrix4D M2 )
{
	//64 mults, 48 adds
	return newMatrix4D(
        newVertex4D(
			(M2.a.x*M1.a.x + M2.a.y*M1.b.x + M2.a.z*M1.c.x + M2.a.w*M1.d.x), 
			(M2.a.x*M1.a.y + M2.a.y*M1.b.y + M2.a.z*M1.c.y + M2.a.w*M1.d.y), 
			(M2.a.x*M1.a.z + M2.a.y*M1.b.z + M2.a.z*M1.c.z + M2.a.w*M1.d.z),
			(M2.a.x*M1.a.w + M2.a.y*M1.b.w + M2.a.z*M1.c.w + M2.a.w*M1.d.w)
		),
        newVertex4D(
			(M2.b.x*M1.a.x + M2.b.y*M1.b.x + M2.b.z*M1.c.x + M2.b.w*M1.d.x), 
			(M2.b.x*M1.a.y + M2.b.y*M1.b.y + M2.b.z*M1.c.y + M2.b.w*M1.d.y), 
			(M2.b.x*M1.a.z + M2.b.y*M1.b.z + M2.b.z*M1.c.z + M2.b.w*M1.d.z),
			(M2.b.x*M1.a.w + M2.b.y*M1.b.w + M2.b.z*M1.c.w + M2.b.w*M1.d.w)
		),
        newVertex4D(
			(M2.c.x*M1.a.x + M2.c.y*M1.b.x + M2.c.z*M1.c.x + M2.c.w*M1.d.x), 
			(M2.c.x*M1.a.y + M2.c.y*M1.b.y + M2.c.z*M1.c.y + M2.c.w*M1.d.y), 
			(M2.c.x*M1.a.z + M2.c.y*M1.b.z + M2.c.z*M1.c.z + M2.c.w*M1.d.z),
			(M2.c.x*M1.a.w + M2.c.y*M1.b.w + M2.c.z*M1.c.w + M2.c.w*M1.d.w)
		),
        newVertex4D(
			(M2.d.x*M1.a.x + M2.d.y*M1.b.x + M2.d.z*M1.c.x + M2.d.w*M1.d.x), 
			(M2.d.x*M1.a.y + M2.d.y*M1.b.y + M2.d.z*M1.c.y + M2.d.w*M1.d.y), 
			(M2.d.x*M1.a.z + M2.d.y*M1.b.z + M2.d.z*M1.c.z + M2.d.w*M1.d.z),
			(M2.d.x*M1.a.w + M2.d.y*M1.b.w + M2.d.z*M1.c.w + M2.d.w*M1.d.w)
		)				
	);
}

Matrix3D mulMatrix3D( Matrix3D M1, Matrix3D M2 )
{
	//27 mults, 24 adds
	return newMatrix3D(
        newVertex3D(
			(M2.a.x*M1.a.x + M2.a.y*M1.b.x + M2.a.z*M1.c.x), 
			(M2.a.x*M1.a.y + M2.a.y*M1.b.y + M2.a.z*M1.c.y), 
			(M2.a.x*M1.a.z + M2.a.y*M1.b.z + M2.a.z*M1.c.z)
		),
        newVertex3D(
			(M2.b.x*M1.a.x + M2.b.y*M1.b.x + M2.b.z*M1.c.x), 
			(M2.b.x*M1.a.y + M2.b.y*M1.b.y + M2.b.z*M1.c.y), 
			(M2.b.x*M1.a.z + M2.b.y*M1.b.z + M2.b.z*M1.c.z)
		),
        newVertex3D(
			(M2.c.x*M1.a.x + M2.c.y*M1.b.x + M2.c.z*M1.c.x), 
			(M2.c.x*M1.a.y + M2.c.y*M1.b.y + M2.c.z*M1.c.y), 
			(M2.c.x*M1.a.z + M2.c.y*M1.b.z + M2.c.z*M1.c.z)
		)
	);
}

void mulMatrix4D( Matrix4D M1, Matrix4D* M2 )
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

void mulMatrix3D( Matrix3D M1, Matrix3D* M2 )
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

Vertex4D mulMatrix4DVertex4D( Matrix4D M, Vertex4D V )
{
	//16 mults; 12 adds
	return newVertex4D( 
		(M.a.x*V.x + M.b.x*V.y + M.c.x*V.z + M.d.x*V.w), 
		(M.a.y*V.x + M.b.y*V.y + M.c.y*V.z + M.d.y*V.w),
		(M.a.z*V.x + M.b.z*V.y + M.c.z*V.z + M.d.z*V.w),
		(M.a.w*V.x + M.b.w*V.y + M.c.w*V.z + M.d.w*V.w)
	);
}

Vertex3D mulMatrix3DVertex3D( Matrix3D M, Vertex3D V )
{
	//16 mults; 12 adds
	return newVertex3D( 
		(M.a.x*V.x + M.b.x*V.y + M.c.x*V.z), 
		(M.a.y*V.x + M.b.y*V.y + M.c.y*V.z),
		(M.a.z*V.x + M.b.z*V.y + M.c.z*V.z)
	);
}

void mulMatrix4DVertex4D( Matrix4D M, Vertex4D* V )
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

void mulMatrix3DVertex3D( Matrix3D M, Vertex3D* V )
{
	//16 mults; 12 adds
	float v1 = M.a.x*V->x + M.b.x*V->y + M.c.x*V->z;
	float v2 = M.a.y*V->x + M.b.y*V->y + M.c.y*V->z;
	float v3 = M.a.z*V->x + M.b.z*V->y + M.c.z*V->z;
	V->x = v1;
	V->y = v2;
	V->z = v3;
}

float lengthVector( Vertex3D V1 ) 
{
	return sqrt(V1.x*V1.x + V1.y*V1.y + V1.z*V1.z);
}

float lengthVector( Vertex4D V1 ) // V1 x V2
{
	return sqrt(V1.x*V1.x + V1.y*V1.y + V1.z*V1.z);
}

Vertex3D crossProduct( Vertex3D V1, Vertex3D V2 ) // V1 x V2
{
	return newVertex3D( (V1.y*V2.z - V1.z*V2.y), (V1.z*V2.x - V1.x*V2.z), (V1.x*V2.y - V1.y*V2.x) );
}

Vertex3D crossProduct( Vertex4D V1, Vertex4D V2 ) // V1 x V2
{
	return newVertex3D( (V1.y*V2.z - V1.z*V2.y), (V1.z*V2.x - V1.x*V2.z), (V1.x*V2.y - V1.y*V2.x) );
}

Vertex3D subVertex3D( Vertex3D V1, Vertex3D V2 ) // V1 - V2
{
	return newVertex3D( (V1.x-V2.x), (V1.y-V2.y), (V1.z-V2.z) );
}

Vertex4D subVertex4D( Vertex4D V1, Vertex4D V2 ) // V1 - V2
{
	return newVertex4D( (V1.x-V2.x), (V1.y-V2.y), (V1.z-V2.z), (V1.w-V2.w) );
}

Vertex3D addVertex3D( Vertex3D V1, Vertex3D V2 ) // V1 + V2
{
	return newVertex3D( (V1.x+V2.x), (V1.y+V2.y), (V1.z+V2.z) );
}

Vertex4D addVertex4D( Vertex4D V1, Vertex4D V2 ) // V1 + V2
{
	return newVertex4D( (V1.x+V2.x), (V1.y+V2.y), (V1.z+V2.z), (V1.w+V2.w) );
}

void writeOutputState( char* fileName, Triangle *allTris, int tCount )
{
	FILE *file; 
	file = fopen(fileName, "w");
	int i;
	
	for(i=0; i<tCount; i++){
		fprintf(file, "Triangle %d \n", i);

		fprintf(file, "v0=(%f %f %f %f) ", allTris[i].v[0].x, allTris[i].v[0].y, allTris[i].v[0].z, allTris[i].v[0].w);
		fprintf(file, "c0=(%f %f %f %f) ", allTris[i].c[0].r, allTris[i].c[0].g, allTris[i].c[0].b, allTris[i].c[0].a);
		//fprintf(file, "n0=(%f %f %f) ", allTris[i].n[0].x, allTris[i].n[0].y, allTris[i].n[0].z);
		fprintf(file, "t0=(%f %f %f)\n",   allTris[i].t[0].x, allTris[i].t[0].y, allTris[i].t[0].z);			
		//fprintf(file, "v_p0=(%f %f) ", allTris[i].v_p[0].x, allTris[i].v_p[0].y);		
		//fprintf(file, "v_s0=(%f %f)\n", allTris[i].v_s[0].x, allTris[i].v_s[0].y);
		
		fprintf(file, "v1=(%f %f %f %f) ", allTris[i].v[1].x, allTris[i].v[1].y, allTris[i].v[1].z, allTris[i].v[1].w);
		fprintf(file, "c1=(%f %f %f %f) ", allTris[i].c[1].r, allTris[i].c[1].g, allTris[i].c[1].b, allTris[i].c[1].a);
		//fprintf(file, "n1=(%f %f %f) ", allTris[i].n[1].x, allTris[i].n[1].y, allTris[i].n[1].z);
		fprintf(file, "t1=(%f %f %f)\n",   allTris[i].t[1].x, allTris[i].t[1].y, allTris[i].t[1].z);
		//fprintf(file, "v_p1=(%f %f) ", allTris[i].v_p[1].x, allTris[i].v_p[1].y);		
		//fprintf(file, "v_s1=(%f %f)\n", allTris[i].v_s[1].x, allTris[i].v_s[1].y);
		
		fprintf(file, "v2=(%f %f %f %f) ", allTris[i].v[2].x, allTris[i].v[2].y, allTris[i].v[2].z, allTris[i].v[2].w);
		fprintf(file, "c2=(%f %f %f %f) ", allTris[i].c[2].r, allTris[i].c[2].g, allTris[i].c[2].b, allTris[i].c[2].a);
		//fprintf(file, "n2=(%f %f %f) ", allTris[i].n[2].x, allTris[i].n[2].y, allTris[i].n[2].z);
		fprintf(file, "t2=(%f %f %f)\n\n", allTris[i].t[2].x, allTris[i].t[2].y, allTris[i].t[2].z);
		//fprintf(file, "v_p2=(%f %f) ", allTris[i].v_p[2].x, allTris[i].v_p[2].y);		
		//fprintf(file, "v_s2=(%f %f)\n\n", allTris[i].v_s[2].x, allTris[i].v_s[2].y);		
	}
	
	fclose(file);
}

void readPPMFile( const char* filename, Color4D **cArr, int *width, int *height )
{
	int i, z, colors, div, mod, idx;
	int* nums[18];
	FILE *file;
	char* line;
	line = (char*)malloc(sizeof(char)*100);
	file = fopen(filename, "r");	
	
	fgets( line, 100, file); // top line; gives file type
	
	fgets( line, 100, file);
	sscanf( line, "%d %d", width, height );

	//printf("width = %d, height = %d\n", *width, *height);
	
	Color4D* out;
	out = *cArr = (Color4D*)malloc(sizeof(Color4D)*(*width)*(*height));
	
	fgets( line, 100, file);	
	sscanf( line, "%d", &colors );
	printf("colors = %d\n", colors);	

	div = ((*height)*(*width)*3)/18;
	mod = ((*height)*(*width)*3)%18;

	idx = 0;
	
	//printf("div = %d, mod = %d\n", div, mod);
	
	for(z = 0; z<div; z++){
	
		fgets( line, 100, file);	
		sscanf( line, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d", 
			&nums[0], &nums[1], &nums[2], &nums[3], &nums[4], &nums[5], &nums[6], &nums[7], &nums[8], 
			&nums[9], &nums[10], &nums[11], &nums[12], &nums[13], &nums[14], &nums[15], &nums[16], &nums[17]);
			
		for(i=0; i<6; i++){
			out[idx].r = 0.0+(int)nums[3*i];
			out[idx].g = 0.0+(int)nums[3*i+1];
			out[idx].b = 0.0+(int)nums[3*i+2];
			idx++;
		}
	}
	
	for(i=0; i<(mod/3); i++){
		out[idx].r = 0.0+(int)nums[3*i];
		out[idx].g = 0.0+(int)nums[3*i+1];
		out[idx].b = 0.0+(int)nums[3*i+2];
		idx++;
	}
		
	//printf("end index = %d\n", idx);	
	//printf("(%f, %f, %f)\n", out[0].r, out[0].g, out[0].b);	
		
	fclose(file);
}

void writePPMFile( const char* filename, Color4D *cArr, int width, int height, int colorDepth ) 
{
	int y, x, idx, r, g, b;
	FILE *file; 
	file = fopen(filename, "w");
	
	fprintf(file, "%s", "P3\n");
	fprintf(file, "%d", width);
	fprintf(file, "%s", " ");
	fprintf(file, "%d", height);
	fprintf(file, "%s", " ");
	fprintf(file, "%d", (colorDepth-1));
	fprintf(file, "%s", "\n\n");
 
	for (y = height-1; y >= 0; y--) {
		for (x = 0; x < width; x++) {
			idx = x + height * y;
			r = (int)cArr[idx].r%colorDepth;
			g = (int)cArr[idx].g%colorDepth;
			b = (int)cArr[idx].b%colorDepth;

			fprintf(file, "%d", r );
			fprintf(file, "%s", " ");
			fprintf(file, "%d", g );
			fprintf(file, "%s", " ");
			fprintf(file, "%d", b ); 
			fprintf(file, "%s", " ");		
			
			//if(idx%18 == 0) fprintf(file, "\n");	
		}
	}
  
	fclose(file);
}

void flipFrame( Color4D *cArr, int width, int height )
{
	Color4D *temp;
	temp = (Color4D*)malloc(sizeof(Color4D)*width);
	int i, j;
	
	if(height%2 == 0){
		for(i=0; i<height/2; i++){
			for(j=0; j<width; j++){
				temp[j] = cArr[i*width+j];
				cArr[i*width+j] = cArr[(height-i-1)*width+j];
				cArr[(height-i-1)*width+j] = temp[j];
			}
		}
	}
	else{
		for(i=0; i<height/2; i++){
			for(j=0; j<width; j++){
				temp[j] = cArr[i*width+j];
				cArr[i*width+j] = cArr[(height-i-1)*width+j];
				cArr[(height-i-1)*width+j] = temp[j];
			}
		}
		for(j=0; j<width/2; j++){
			temp[j] = cArr[i*width+j];
			cArr[i*width+j] = cArr[i*width+(width/2 + j + 1)];
			cArr[i*width+(width/2 + j + 1)] = temp[j];			
		}
	}
		
}

void printAppConfig( AppConfig ap )
{
	printf("radius = %f\n", ap.radius);
	printf("divisions = %d\n", ap.divisions);
	printf("do_parallel = %d\n", ap.do_parallel);
	printf("do_perspective_correct = %d\n", ap.do_perspective_correct);
	printf("do_texture = %d\n", ap.do_texture);
	printf("do_depth_test = %d\n", ap.do_depth_test);
	printf("do_area_test = %d\n", ap.do_area_test);
	printf("do_geom_test = %d\n", ap.do_geom_test);
	printf("output file = %s\n\n", ap.output_file);
}

void printMatrix4D( Matrix4D M )
{
	printf("(\t%.3f\t%.3f\t%.3f\t%.3f\t)\n",   M.a.x, M.b.x, M.c.x, M.d.x);
	printf("(\t%.3f\t%.3f\t%.3f\t%.3f\t)\n",   M.a.y, M.b.y, M.c.y, M.d.y);
	printf("(\t%.3f\t%.3f\t%.3f\t%.3f\t)\n",   M.a.z, M.b.z, M.c.z, M.d.z);
	printf("(\t%.3f\t%.3f\t%.3f\t%.3f\t)\n\n", M.a.w, M.b.w, M.c.w, M.d.w);
}

void printMatrix3D( Matrix3D M )
{
	printf("(\t%.3f\t%.3f\t%.3f\t)\n",   M.a.x, M.b.x, M.c.x);
	printf("(\t%.3f\t%.3f\t%.3f\t)\n",   M.a.y, M.b.y, M.c.y);
	printf("(\t%.3f\t%.3f\t%.3f\t)\n",   M.a.z, M.b.z, M.c.z);
}

void printVertex4D( Vertex4D V ){ printf("(\t%.3f\t%.3f\t%.3f\t%.3f\t)\n",   V.x, V.y, V.z, V.w); }
void printVertex3D( Vertex3D V ){ printf("(\t%.3f\t%.3f\t%.3f\t)\n",   V.x, V.y, V.z); }

float getTriangleArea( Triangle T )
{
	//10 mul, 11 add
	Vertex3D cp = crossProduct( subVertex4D( T.v[2], T.v[0] ), subVertex4D( T.v[2], T.v[1] ) );
	if(cp.z < 0.0) return 0.5*sqrt( cp.x*cp.x + cp.y*cp.y + cp.z*cp.z );
	else return -0.5*sqrt( cp.x*cp.x + cp.y*cp.y + cp.z*cp.z );
}

float getTriangleAreaTest( float a, float b, float c, float d, float e, float f, float g, float h, float i )
{
	//10 mul, 11 add
	Vertex3D cp = crossProduct( subVertex4D( newVertex4D(g, h, i, 0), newVertex4D(a, b, c, 0)), subVertex4D( newVertex4D(g, h, i, 0), newVertex4D(d, e, f, 0) ) );
	return 0.5*sqrt( cp.x*cp.x + cp.y*cp.y + cp.z*cp.z );
}

unsigned short squareSubPixel( unsigned short pixel, unsigned short subpixel )
{
	//4 mul, 2 add
	unsigned short div = subpixel >> 13;
	return unsigned short(pixel*pixel + 2*pixel*div + div*div); 
}

Matrix4D quickMatrix4D( float a, float b, float c, float d,
						float e, float f, float g, float h,
						float i, float j, float k, float l,
						float m, float n, float o, float p)
{
	return newMatrix4D(
		newVertex4D(a, e, i, m),
		newVertex4D(b, f, j, n),
		newVertex4D(c, g, k, o),
		newVertex4D(d, h, l, p)
	);
}

Matrix3D quickMatrix3D( float a, float b, float c,
						float d, float e, float f,
						float g, float h, float i)
{
	return newMatrix3D(
		newVertex3D(a, d, g),
		newVertex3D(b, e, h),
		newVertex3D(c, f, i)
	);
}

Matrix2D quickMatrix2D( float a, float b,
						float c, float d)
{
	return newMatrix2D(
		newVertex2D(a, c),
		newVertex2D(b, d)
	);
}
			
float dCos( float angle ){ return cos(angle*CONVERT); }					
float dSin( float angle ){ return sin(angle*CONVERT); }	

void bitFieldOn( unsigned short *bitField, int index )
{	
	int off = index >> 4; //int divide by 16
	bitField[off] = bitField[off] | (1UL << (index & 15UL));
}

void bitFieldOff( unsigned short *bitField, int index )
{
	int off = index >> 4; //int divide by 16
	bitField[off] = bitField[off] & ~(1UL << (index & 15UL));
}

int bitFieldGet( unsigned short *bitField, int index )
{
	return ((bitField[index >> 4] & (1UL << (index & 15UL))) != 0);
}

/* assumes that all vertices have been frustum culled and projected into 2048*2048 viewport (therefore all values are [0, 2048]) */
void snapCoord( float input, short *px, short *sub_px, int grid_bits )
{
	input *= (1UL << grid_bits);
	int tempInt = (int)input;
	*px = tempInt >> grid_bits;
	*sub_px = tempInt & ((1UL << grid_bits)-1);
}

float unSnapCoord( short px, short sub_px, int grid_bits ){ return float(px+(float(sub_px)/(1UL << grid_bits))); }

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}

int triTypeSort( Vertex4D *v1, Vertex4D *v2, Vertex4D *v3, Color4D *c1, Color4D *c2, Color4D *c3, Vertex3D *t1, Vertex3D *t2, Vertex3D *t3 )
{
	int tri_case;
	Vertex4D v_t = newVertex4D(0, 0, 0, 0);
	Vertex4D *vt;
	vt = &v_t;
	Color4D c_t = newColor4D(0, 0, 0, 0);
	Color4D *ct;
	ct = &c_t;
	Vertex3D t_t = newVertex3D(0, 0, 0);
	Vertex3D *tt;
	tt = &t_t;
	
	minVertY( v1, c1, t1, v2, c2, t2 );
	tri_case = minVertY( v1, c1, t1, v3, c3, t3 );

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
		minVertY( v2, c2, t2, v3, c3, t3 ); 
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

int minVertY( Vertex4D *v1, Color4D *c1, Vertex3D *t1, Vertex4D *v2, Color4D *c2, Vertex3D *t2 )
{
	Vertex4D v_t = newVertex4D(0, 0, 0, 0);
	Vertex4D *vt;
	vt = &v_t;
	Color4D c_t = newColor4D(0, 0, 0, 0);
	Color4D *ct;
	ct = &c_t;
	Vertex3D t_t = newVertex3D(0, 0, 0);
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

void fineRast( Triangle* tri_culled_d, Color4D* frame_buffer_d, Float1D* depth_buffer_d, Color4D* tex_buffer_d, 
				int screen_width, int tex_width, int DO_TEXTURE, int PERSPECTIVE_CORRECT, int DO_DEPTH_TEST ) 
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
	int tri_case, x, y; //1 = left higher; 2 = right higher; 3 = top flat; 4 = bottom flat

	v1 = &tri_culled_d->v[0];
	v2 = &tri_culled_d->v[1];
	v3 = &tri_culled_d->v[2];
	c1 = &tri_culled_d->c[0];
	c2 = &tri_culled_d->c[1];
	c3 = &tri_culled_d->c[2];
	t1 = &tri_culled_d->t[0];
	t2 = &tri_culled_d->t[1];
	t3 = &tri_culled_d->t[2];

	/* sort verts by y */
	tri_case = triTypeSort( v1, v2, v3, c1, c2, c3, t1, t2, t3 );

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
					integrateFragment( frame_buffer_d, depth_buffer_d, tex_buffer_d, x, (int)curr_vy, screen_width, tex_width,
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
					integrateFragment( frame_buffer_d, depth_buffer_d, tex_buffer_d, x, (int)curr_vy, screen_width, tex_width,
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
		}
		last_l_vy = curr_vy;
		curr_vy += 1.0;
		diff_vy = 1.0;
	}
}

void integrateFragment( Color4D* f_buf_d, Float1D* d_buf_d, Color4D* t_buf_d, int x_coord, int y_coord, int screen_width, int tex_width,
						float curr_z, float curr_cr, float curr_cg, float curr_cb, float curr_ca, float curr_tx, float curr_ty, int DO_TEXTURE, int DO_DEPTH_TEST )
{
	int idx = x_coord + y_coord*screen_width;
	
	// depth test; if current depth is greater than value in depth buffer (and therefore this 
	// fragment is farther away than curren closest), do nothing
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

void buildCheckerBoard( int textureSize, int tileSize, Color4D **tex_buffer )
{
	int i, j, indx;
	
	Color4D *tbuf;
	tbuf = *tex_buffer = (Color4D *)malloc(sizeof(Color4D)*textureSize*textureSize);

	Color4D black={0x0, 0x0, 0x0, 0x0};
	Color4D white={0xFF, 0xFF, 0xFF, 0xFF};
	
	indx = 0;
	for (j=0;j<textureSize;j++)
	{
		Color4D *even, *odd;
		//if ((j&1) == 0)  
		if ((j/tileSize)%2 == 0) 
		{
			even = &black;
			odd = &white;
		}
		else
		{
			even = &white;
			odd = &black;
		}

		for (i=0;i<textureSize;i++)
		{
			if ((i/tileSize)%2 == 0) 
				tbuf[indx] = *even;
			else
				tbuf[indx] = *odd;
			indx++;
		}
	}
}

void parseArgs(int argc, char** argv, AppConfig &ap)
{
	for(size_t i = 1; i < argc; i += 2)
	{
		if (i + 1 != argc)
		{
			
			if (strcmp(argv[i], "--radius") == 0){ ap.radius = atoi(argv[i + 1]); } 
			else if (strcmp(argv[i], "--divisions") == 0){ ap.divisions = atof(argv[i + 1]); } 
			else if (strcmp(argv[i], "--parallel") == 0){ ap.do_parallel = atoi(argv[i + 1]); } 
			else if (strcmp(argv[i], "--perspective") == 0){ ap.do_perspective_correct = atoi(argv[i + 1]); } 
			else if (strcmp(argv[i], "--texture") == 0){ ap.do_texture = atoi(argv[i + 1]); } 
			else if (strcmp(argv[i], "--depth") == 0){ ap.do_depth_test = atoi(argv[i + 1]); } 
			else if (strcmp(argv[i], "--doareatest") == 0){ ap.do_area_test = atoi(argv[i + 1]); } 
			else if (strcmp(argv[i], "--dogeomtest") == 0){ ap.do_geom_test = atoi(argv[i + 1]); } 
			else if (strcmp(argv[i], "--output") == 0){ 
				char *first = argv[i + 1];
				char *second = ".ppm";
				char *both = (char *)malloc(strlen(first) + strlen(second) + 1);
				strcpy(both, first);
				strcat(both, second);
				ap.output_file = both;
			} 
			else {
				printf("Not enough or invalid arguments, please try again.\n");
				exit(0);
			}
		}
	}
}

AppConfig initialize(int argc, char** argv)
{
	AppConfig ap;
	ap.radius 					= DEFAULT_RADIUS;
	ap.divisions 				= DEFAULT_DIVISIONS;
	ap.do_parallel				= DEFAULT_PARALLEL;
	ap.do_perspective_correct 	= DEFAULT_DO_PERSPECTIVE_CORRECT;
	ap.do_texture				= DEFAULT_DO_TEXTURE;
	ap.do_depth_test			= DEFAULT_DO_DEPTH_TEST;
	ap.do_area_test				= DEFAULT_DO_AREA_TEST;
	ap.do_geom_test				= DEFAULT_DO_GEOM_TEST;
	ap.output_file				= "output.ppm";

	parseArgs(argc, argv, ap);
	return ap;
}
