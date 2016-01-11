#include <stdio.h>
#include <stdlib.h>
#include "./p3.h"
#define N 6

int main(int argc, char *argv[]){
  int task[4] = {100,50,10,200};
  if (argc != 3){
     printf("Exiting. Thread or Size of task not provided.\n");
     printf("Please see README for instructions to run the program. \nSize of Task: 1-  Large : 2- Medium : 3- Small 4- Mixed\n");
     exit(1);
  }

  int M = atoi(argv[1]);
  int T = atoi(argv[2]);
  
  if(T>4 || T<1) {
    printf("Exiting.Incorrect Input.\n");
    printf("Suitable Values Are: 1 for Large. 2 For Medium. 3 For Small. 4 for Mixed Inputs \n");
    exit(1);
  } 
  int ret = call_sched(N, M, task[T-1]);
  if (ret != 1) {
     printf("CUDA call failed\n");
  }
  return 0;
}
