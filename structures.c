
/*////////////////////////////*/						
/* STRUCTURE FUNCTION LIBRARY */					
/*////////////////////////////*/	

typedef struct{ short x, y; } Vertex2D_px;
typedef struct{ short x, y; } Vertex2D_sub_px;

typedef struct{ float x, y; } Vertex2D;
typedef struct{ float x, y, z; } Vertex3D;
typedef struct{ float x, y, z, w; } Vertex4D;

typedef struct{ float r, g, b; } Color3D;
typedef struct{ float r, g, b, a; } Color4D;

//each Vertex represents a COLUMN of the matrix
typedef struct{ Vertex4D a, b, c, d; } Matrix4D;
typedef struct{ Vertex3D a, b, c; } Matrix3D;
typedef struct{ Vertex2D a, b; } Matrix2D;

typedef struct{ float f; } Float1D;
typedef struct{ int i; } Int1D;

typedef struct{ Float1D F; } FloatFloat1D;
typedef struct{ Int1D I; } IntInt1D;

typedef struct{ 
	Vertex4D 			v[3];
	Vertex3D 			t[3];
	Color4D  			c[3];
	Vertex3D 			n[3];
} Triangle;

// Struct to hold the app's state
typedef struct{
	float radius;
	unsigned int divisions;
	unsigned int do_perspective_correct;
	unsigned int do_texture;
	unsigned int do_depth_test;
	unsigned int do_parallel;
	unsigned int do_area_test;
	unsigned int do_geom_test;
	char* output_file;
} AppConfig;