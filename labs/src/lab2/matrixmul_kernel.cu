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
	//allocate answer tile Pt to shared mem
	__shared__ float Pt[TILE_DIM * (TILE_DIM + 1)]; //allocate 32x33 matrix in mem
	unsigned int bRow, bCol;
	bRow = blockIdx.y;
	bCol = blockIdx.x;

	float Pvalue = 0;

	// //get index within sub matrix
	unsigned int tRow, tCol;
	tRow = threadIdx.y;
	tCol = threadIdx.x;

	//get common dimension
	unsigned int comm_dim = (M.width);

	//block row offset
	unsigned int M_block_row_offset = bRow * M.width * TILE_DIM;
	unsigned int M_thread_row_offset = tRow * M.width;
	unsigned int N_thread_row_offset = tRow * N.width;
	unsigned int N_thread_col_offset = bCol * blockDim.x;

	//loop through all rows and colomns of tiles of M and N to compute this value in the output Tile
	for (int i = 0; i <  comm_dim/TILE_DIM; ++i)
	{
		//declare our sub matrices into shared mem
		__shared__ float Mt[TILE_DIM * (TILE_DIM + 1)];
		__shared__ float Nt[TILE_DIM * (TILE_DIM + 1)];

		//move data from our matrices in global mem in a coalesced manner
		//each thread in warp should load in a row
		*(Mt + tRow * blockDim.x + tCol) = *(M.elements + M_block_row_offset + M_thread_row_offset + i * blockDim.y + tCol);
		*(Nt + tRow * blockDim.x + tCol) = *(N.elements + i * N.width * TILE_DIM + N_thread_row_offset + N_thread_col_offset + tCol);


	} 
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
