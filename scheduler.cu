#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "./p3.h"



__device__  int F = 0;

__device__ uint get_smid(void) {
  uint ret;
  asm("mov.u32 %0, %smid;" : "=r"(ret) );
  return ret;
}

//Sid: Function to be called on device for addition of the elements from queue[3] to queue[103] and so on
extern "C"
__device__ void* calcValue(void *arg) {
      int i = 0;
      int sum = 0;
      int *args = (int*)arg;
      int len = args[0];
      while( i < len) {
          sum += args[i];
          i++;
      }
      return (void*)sum;
}
/*
      int size = queues[(sm*104)+1].val;
      int thread_no = queues[624].val;

      int steps = size/thread_no;
      int rem = size-(steps*thread_no);
      int i = 0;
      int start_index = (sm*104)+3;

      if (threadIdx.x < rem) {
         steps+=1;
      }
      if (threadIdx.x < size) {
         for (i = 0; i<steps; i++) {
             atomicAdd(&queues[(sm*104)+2].val, queues[start_index+(i*thread_no)+threadIdx.x].val);
             queues[start_index+(i*thread_no)+threadIdx.x].Done = 1;
         }
      }
      __syncthreads();
      //Sid:Check that all tasks for the SM are done
      for( i = start_index; i<=(start_index+100); i++){
         if (queues[i].Done = 0) {
            queues[sm*104].Done = 0;
            break;
         } else {
            queues[sm*104].Done = 1;
         }
      }*/

//Sid: Add Static Function Pointers to device functions
__device__ op_func_t p_calc_func = calcValue;

//Sid: Schedule the tasks based on the task_done flag
__global__ void scheduler(taskQueue_t *queues, int *arg_d, int *done_t, int *T_d) {
   int sm = get_smid();
   void *sum;
   int index;
   index = (sm*queues[threadIdx.x].M)+ threadIdx.x;
   while (F < (*T_d -1)) {
      if (threadIdx.x < queues[index].M) {
         if (queues[index].taskDone == 0) {
            //Sid: Call function pointer
            sum = (*(queues[index].func))( arg_d);
            queues[index].val = (int)sum;
            queues[index].taskDone = 1;
            done_t[index]+=1;
            //      queues[index].func = NULL;
            atomicAdd(&F,1);
         }
      }
   }
}

//Sid:Add tasks
int taskAdd(void *(*func) (void *), void *arg, int sm) {
  int index = sm;
  int M; 
  M = ((int*)arg)[1];
  int sm_no = index/M;
  op_func_t h_calc_func;
  //Sid: Copy device function pointer to host side
  cudaMallocHost(&h_calc_func, sizeof(op_func_t));
  cudaStream_t S;
  cudaStreamCreate(&S);
  cudaMemcpyFromSymbolAsync( &h_calc_func, p_calc_func, sizeof( op_func_t ), 0, cudaMemcpyDeviceToHost, S );
  //Sid: func = h_calc_func is for addition.This implementation is limited to Addition only. We can extend it to other functionalities as well
  queue[index].func =  h_calc_func;
  queue[index].sm = sm_no;
  queue[index].M = M;
  queue[index].taskDone = 0;
  gettimeofday(&queue[index].start,NULL);
  gettimeofday(&queue[index].end,NULL);
  //Sid: Initialize Result to 0
  queue[index].val = 0;
 
  return SUCCESS;
}

//Sid: Check Task Done
int taskDone(int taskId) {
    return queue[taskId].taskDone;
}
//Sid:Below Code for calculating time interval Taken from Homework3 TfIDf program
long calcDiffTime(struct timeval* strtTime, struct timeval* endTime) {

return( endTime->tv_sec*1000000 + endTime->tv_usec - strtTime->tv_sec*1000000 - strtTime->tv_usec );
}

extern "C" int call_sched(int N, int M, int task) {
  int arg[3];
  int *args;
  int *arg_d;
  int *T_d;
  int *done;
  int *done_t;
  int *count;
  int i = 0;
  int j = 0;
  int size = N*M*(sizeof(taskQueue_t));
  int size1 = task*(sizeof(int));
  //Sid: Device Queue
  taskQueue_t *dev_queue;
  long DiffTime;
  int counter = 0;
  int wait = 0;
  int race_check = 0;
  int current_index = 0;
  int sm_task_done = 0;
  int task_left = 0;

  count = (int*)malloc(M*N*sizeof(int));
  for (i = 0; i<M*N; i++) {
      count[i] = 0;
  }
  //Sid: Create event to synchronize between streams
  cudaEvent_t event1;
  cudaEventCreate(&event1);
  //Sid: Create two streams. Code referenced from http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution
  cudaStream_t S0, S1;
  cudaStreamCreate(&S0); 
  cudaStreamCreate(&S1); 

  //Sid: Allocate pinned memory on the host side for queue
  cudaMallocHost(&queue, (N * M) * sizeof(taskQueue_t));
  
  //Sid: Allocate memory for arguments
  cudaMallocHost(&args, task*sizeof(int));
  arg[0] = N;
  arg[1] = M;
  arg[2] = task;
  
  //Sid: Allocate memory for done[] on host side, of size equal to number of threads and initialize
  cudaMallocHost(&done, M*N*sizeof(int));
  for (i=0; i<M*N; i++) {
      done[i] = 0;
  }

  //Sid: Allocate memory for done on GPU of size equal to number of threads
  cudaMalloc((void**)&done_t, M*N*sizeof(int));
  cudaMemcpy(done_t, done, M*N*sizeof(int), cudaMemcpyHostToDevice);
  //Sid: Allocate memory for task size
  cudaMalloc((void **)&T_d, sizeof(int));
  cudaMemcpy(T_d, &task, sizeof(int), cudaMemcpyHostToDevice);

  //Sid: Initialize task queue
  for (i = 0; i<N*M; i++) {
     queue[i].func = 0;
  }

  cudaMalloc((void**)&dev_queue, size);
  //Sid: Add M*N tasks to the queues for each of the SMs
  for (i = 0; i<(N*M); i++) {
     taskAdd(calcValue,&arg, i);
  }
  
  //Sid: Set the first element of args to length of task
  args[0] = task-1;
  //Sid: Allocate arguments and assign values of size task. So argument will be of size task always in this scenario. Can be changed later to vary it.
  for (i=1;i<task;i++) {
     args[i] = i-1;
  }
  //Sid: Copy the arguments to the GPU
  cudaMalloc((void**)&arg_d, size1);
  cudaMemcpy(arg_d, args, size1, cudaMemcpyHostToDevice);
  
  //Sid: Initial load of tasks copied to GPU
  cudaMemcpy(dev_queue, queue, size, cudaMemcpyHostToDevice);
  scheduler<<<N,M,0,S0>>>(dev_queue, arg_d, done_t, T_d);

  //Sid: Polling Loop : Wait here and check that all tasks are done
  while (wait == 0) {
     //Sid: Update the local queue.
     if (cudaSuccess != cudaMemcpyAsync(queue, dev_queue, size, cudaMemcpyDeviceToHost,S1)) {
         printf ("CUDA MemCpy Async periodic Update from Device to Host Failed\n");
         exit(0);
     }
     for (i = 0; i <N; i++) {
       if (counter >= task) {
          wait = 1;
          break;
       } 
       for (j = 0; j<M; j++) {
          //Sid: If RaceCheck has not been updated back to 0 for a long time then break
          if (race_check > task*100) {
             printf ("\n>>>>>>>Indications of program probably being stuck in a race condition. Exiting Gracefully!! \n");
             exit(1);
          }
          race_check++;
          if (taskDone(current_index) == 1) {
             gettimeofday(&queue[current_index].end, NULL);
             DiffTime = calcDiffTime(&(queue[current_index].start), &(queue[current_index].end));
             count[current_index]+=1;
             counter+=1;
             sm_task_done+=1;
             race_check = 0;
             printf("SM No: %d Thread No. %d Completed!\nTime Taken: %ld Results: %d\n_________________________\n",i,current_index,DiffTime,queue[current_index].val);
             if (counter >= task) {
                wait = 1;
                break;
             }
          }
          current_index++;
       }
       //Sid: If more tasks are remaining add a few more tasks to this SM if all the threads in this SM are done
       if (counter < task ) {
          if (sm_task_done == M) {
             current_index = (i*M);
             task_left = task - counter;
             for (j=0; j<M; j++) {
                if (j < task_left) {
                   taskAdd(calcValue,&arg, current_index);
                }
                current_index++;
             }
          }
       } else {
          wait = 1;
          break;
       }
       //Sid: Reset the value of SM_task_done to check for the next SM
       sm_task_done = 0;
    }
    current_index = 0;
    //Sid: Copy the tasks to the GPU
    cudaMemcpyAsync(dev_queue, queue, size, cudaMemcpyHostToDevice,S1);
    cudaEventRecord(event1 ,S1);
    cudaStreamSynchronize(S1);     
    cudaStreamWaitEvent( S0 , event1 , 0);
  }
  cudaDeviceSynchronize();
  //Sid:Copy back results to host when all SM are finished
  if (cudaSuccess != cudaMemcpy(queue, dev_queue, size, cudaMemcpyDeviceToHost)) {
      printf ("CUDA MemCpy Async From Device to Host Failed\n");
      exit(0);
  }
  cudaMemcpy(done, done_t, M*N*sizeof(int), cudaMemcpyDeviceToHost);
  printf("\nN X M Matrix\n");
  current_index = 0;
  printf("Done[]|Count[]\n");
  for (i = 0; i< N; i++) {
    for (j=0; j<M; j++) {
        printf("%d|%d ",count[current_index],done[current_index]);
        current_index++;
    }
    printf("\n");
  }
  //Sid: Cleanup
  cudaFree(dev_queue);
  return SUCCESS;
}
