/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////

#define TILE_DIM 32 //would like to get this # from block size in DEVICE function

// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
	//allocate input and answer tiles to shared mem with padding to avoid bank conflicts within warps
	__shared__ float Pt[TILE_DIM][TILE_DIM + 1]; 
	__shared__ float Mt[TILE_DIM][TILE_DIM + 1];
	__shared__ float Nt[TILE_DIM][TILE_DIM + 1];
	memset(Nt, 0, TILE_DIM * (TILE_DIM + 1));
	memset(Mt, 0, TILE_DIM * (TILE_DIM + 1));
	float Pvalue = 0;
	//get block coords in grid
	unsigned int bRow, bCol;
	bRow = blockIdx.y;
	bCol = blockIdx.x;
	// get thread coords in block
	unsigned int tRow, tCol;
	tRow = threadIdx.y;
	tCol = threadIdx.x;

	//index matches rows of Pd & Md
	unsigned int row = bRow * TILE_DIM + tRow;
	//index matches cols of pd & Nd
	unsigned int col = bCol * TILE_DIM + tCol;

	//get how many tiles needed to step across input matrices based on their common dimension (width {cols} of M or height {rows} of N)
	unsigned int steps = M.width / TILE_DIM;
	if (M.width % TILE_DIM){
		steps++;
	} 

	//loop through all rows and colomns of tiles of M and N to compute this value in the output Tile
	for (int i = 0; i <  steps; i++) {
		//move data from our matric(es in global mem in a coalesced manner
		//each thread in warp should load in a row
		Mt[tRow][tCol] = M.elements[row + i * TILE_DIM + tCol];
		Nt[tRow][tCol] = N.elements[(i * N.width * TILE_DIM) + (tRow * N.width) + col];
		__syncthreads();		

		for (int j = 0; j < TILE_DIM; ++j) {
			Pvalue += Mt[tRow][j] * Nt[j][tCol];
		}
		__syncthreads();		
	} 
	//write back
	Pt[tRow][tCol] = Pvalue;
	P.elements[row * P.width + col]= Pt[tRow][tCol];
	return;
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
