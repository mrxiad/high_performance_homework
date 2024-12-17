#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
 
#define THREAD_NUMBER 2
 
int retval_hello1 = 1;
int retval_hello2 = 2;
 
void* hello1(void* arg){
    char* hello_str = (char *)arg;
    sleep(1);
    printf("%s\n", hello_str);
    pthread_exit(&retval_hello1);
}
 
void* hello2(void* arg){
    char* hello_str = (char *)arg;
    sleep(2);
    printf("%s\n", hello_str);
    pthread_exit(&retval_hello2);
}
 
int main()
{
    int retval;
    int *retval_hello[2];
 
    pthread_t pt[2];
    const char* arg[THREAD_NUMBER];
    arg[0] = "hello world from thread1.";
    arg[1] = "hello world from thread2.";
    printf("begin to create threads....\n");
 
    retval = pthread_create(&pt[0], NULL, hello1, (void*)arg[0]);
    if(retval != 0){
        printf("pthread_create error.");
        exit(1);
    }
    retval = pthread_create(&pt[1], NULL, hello2, (void*)arg[1]);
    if(retval != 0){
        printf("pthread_create error.");
        exit(1);
    }
 
    printf("now, the main thread returns.\n");
    printf("main thread begins to wait threads.\n");
    for(int i=0;i<THREAD_NUMBER;i++){
        retval = pthread_join(pt[i], (void **)&retval_hello[i]);
        if(retval != 0){
            printf("pthread_join error");
            exit(1);
        }else{
            printf("return value is %d\n", *retval_hello[i]);
        }
    }
    return 0;
}