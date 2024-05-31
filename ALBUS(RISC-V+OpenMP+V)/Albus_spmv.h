#include<iostream>
#include<stdio.h>
#include<math.h>
#include<time.h>
#include<omp.h>

#include <riscv-vector.h>

#include<cstring>
#include<sys/time.h>
#include<stdlib.h>
using namespace std;

#define INT int
#define DOU double

inline DOU RV_fast2(INT start1,INT num,INT * __restrict row_ptr,INT * __restrict col_idx,DOU * __restrict mtx_val,DOU * __restrict vec_val)
{
	DOU answer = 0;
	INT end1 = start1 + num;
	while (start1 < end1) {
		answer += mtx_val[start1] * vec_val[col_idx[start1]];
		start1++;
	}
	return answer;
}

inline DOU RV_fast1(INT start1,INT num,INT * __restrict row_ptr,INT * __restrict col_idx,DOU * __restrict mtx_val,DOU * __restrict vec_val)
{
	constexpr size_t vl = 8;
	INT end1 = start1 + num;
	DOU answer = 0;
	DOU temp[8]{};
	
	vfloat64m4_t v_1, v_2;
	vfloat64m4_t v_summ = vle_v_f64m4(temp, vl);
	while (num > vl) {
		temp[0] = vec_val[col_idx[start1]];
		temp[1] = vec_val[col_idx[start1 + 1]];
		temp[2] = vec_val[col_idx[start1 + 2]];
		temp[3] = vec_val[col_idx[start1 + 3]];
		temp[4] = vec_val[col_idx[start1 + 4]];
		temp[5] = vec_val[col_idx[start1 + 5]];
		temp[6] = vec_val[col_idx[start1 + 6]];
		temp[7] = vec_val[col_idx[start1 + 7]];
		v_1 = vle_v_f64m4(mtx_val + start1, vl);
		v_2 = vle_v_f64m4(temp, vl);
		v_summ = vfmacc_vv_f64m4(v_summ, v_1, v_2, vl);
		start1 += vl;
		num -= vl;
	}
	vfloat64m1_t v_res = vfmv_v_f_f64m1(0.0, vsetvlmax_e64m1());
	v_res = vfredsum_vs_f64m4_f64m1(v_res, v_summ, v_res, vl);
	vse_v_f64m1(&answer, v_res, 1);
	while (start1 < end1) {
		answer += mtx_val[start1] * vec_val[col_idx[start1]];
		start1++;
	}
	return answer;
}

inline DOU calculation(INT start1,INT num,INT * __restrict row_ptr,INT * __restrict col_idx,DOU * __restrict mtx_val,DOU * __restrict vec_val)
{
	// vsetvlmax_e64m4() == 8
	if (num >= 8)
		return RV_fast1(start1,num,row_ptr,col_idx,mtx_val,vec_val);
	else
		return RV_fast2(start1,num,row_ptr,col_idx,mtx_val,vec_val);
}

inline void thread_block(INT thread_id,INT start,INT end,INT start2,INT end2,INT * __restrict row_ptr,INT * __restrict col_idx,DOU * __restrict mtx_val,DOU * __restrict mtx_ans,DOU * __restrict mid_ans,DOU * __restrict vec_val)
{

        register INT start1,end1,num,Thread,i;
	register DOU sum;
	switch(start < end)
	{
		case true: {
			mtx_ans[start]  = 0.0;
			mtx_ans[end]    = 0.0;
        		start1          = row_ptr[start] + start2;
			start++;
        		end1            = row_ptr[start];
        		num             = end1 - start1;
        		Thread          = thread_id<<1;
        		mid_ans[Thread] = calculation(start1,num,row_ptr,col_idx,mtx_val,vec_val);
			start1 = end1;
	
			#pragma simd
        		for(i=start;i<end;++i)
        		{
               		 	end1       = row_ptr[i+1];
                		num        = end1 - start1;
                		sum        = calculation(start1,num,row_ptr,col_idx,mtx_val,vec_val);
				mtx_ans[i] = sum;
				start1 = end1;
        		}
			start1 = row_ptr[end];
        		end1   = start1 + end2;
        		mid_ans[Thread | 1] = calculation(start1,end2,row_ptr,col_idx,mtx_val,vec_val);
			return ;
		}
		default : {
			mtx_ans[start]      = 0.0;
			Thread              = thread_id<<1;
			start1              = row_ptr[start] + start2;
			num                 = end2 - start2;
			mid_ans[Thread]     = calculation(start1,num,row_ptr,col_idx,mtx_val,vec_val);
			mid_ans[Thread | 1] = 0.0;
			return ;
		}
	}
}

inline INT binary_search(INT *&row_ptr,INT num,INT end)
{
        INT l,r,h,t=0;
        l=0,r=end;
        while(l<=r)
        {
                h = (l+r)>>1;
                if(row_ptr[h]>=num)
                {
                        r=h-1;
                }
                else
                {
                        l=h+1;
                        t=h;
                }
        }
        return t;
}

inline void albus_balance(INT *&row_ptr,INT *&par_set,INT *&start,INT *&end,INT *&start1,INT *&end1,DOU *&mid_ans,INT thread_nums)
{
        register int tmp;
	start[0]            = 0;
	start1[0]           = 0;
        end[thread_nums-1]  = par_set[0];
	end1[thread_nums-1] = 0;
        INT tt=par_set[2]/thread_nums;
        for(INT i=1;i<thread_nums;i++)
        {
                tmp=tt*i;
                start[i]  = binary_search(row_ptr,tmp,par_set[0]);
		start1[i] = tmp - row_ptr[start[i]];
                end[i-1]  = start[i];
		end1[i-1] = start1[i];
        }
}

inline void SPMV_DOU(INT * __restrict row_ptr,INT * __restrict col_idx,DOU * __restrict mtx_val,INT * __restrict par_set,DOU * __restrict mtx_ans,DOU * __restrict vec_val,INT * __restrict start,INT * __restrict end,INT * __restrict start1,INT * __restrict end1,DOU * __restrict mid_ans, INT thread_nums)
{
        register INT i;
        #pragma omp parallel private(i)
        {
                #pragma omp for schedule(static) nowait
		for(i=0;i<thread_nums;++i)
                {
                        thread_block(i,start[i],end[i],start1[i],end1[i],row_ptr,col_idx,mtx_val,mtx_ans,mid_ans,vec_val);
                }
        }
        mtx_ans[0] = mid_ans[0];
        INT sub;
	#pragma unroll(32)
        for(i=1;i<thread_nums;++i)
        {
		sub = i<<1;
		register INT tmp1 = start[i];
		register INT tmp2 = end[i-1];
                if(tmp1 == tmp2)
                {
                        mtx_ans[tmp1] += (mid_ans[sub-1] + mid_ans[sub]);
                }
                else
                {
                        mtx_ans[tmp1] += mid_ans[sub];
                        mtx_ans[tmp2] += mid_ans[sub-1];
                }
        }
}
