																																
 SoftRast: Software Rasterization on Parallel Hardware, v1.0																	
 by Daniel J. Chase																											
																																
 Usage instructions:	
 Requires a NVIDIA graphics card which supports CUDA.  
 Compile using nvcc.exe and run the softRast executable. The following flags are all optional:								
																																
 --radius (float)																												
	 	Sets the radius value and therefore the area of the test sphere to be rendered. Default = 1.5.							
 --divisions (int)																											
		Sets the number of divisions (both in phi and theta) and therefore the number of vertices in the test sphere.			
		Vertex count is O(n^2) of the number of divisions. Default = 15.														
 --parallel (1/0)																												
		Controls whether the rendering is handled by the CPU or the GPU. Default = 1.											
 --perspective (1/0)																											
		Turns perspective-correction on and off. Default = 1.																	
 --texture (1/0)																												
		Turns texture mapping on and off. Default = 0.																			
 --depth (1/0)																												
		Turns the depth buffer on and off. Default = 1.																			
 --doareatest (1/0)																											
		Performs an area sweep, testing from radius = 0.1 to radius = 3.0. Default number of divisions may be overridden.		
 --dogeomtest (1/0)																											
		Performs a vertex sweep, testing from divisions = 1 to divisions = 30. Default radius value may be overridden.			
 --output (char*)																												
		Sets the name of the output file for the rendered image. Do not include a file extension; delivered file will be in *.ppm format, which may be converted to *.png or similar using ImageMagick. Output file will not be generated during an area or geometry test. Default = "output".																	
																																
