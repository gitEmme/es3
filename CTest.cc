#include <iostream>
#include <chrono>
#include <unistd.h>
#include"tensor.h"

using namespace Tensor;

std::ostream & operator << (std::ostream& out, Index_Set<>) { return out; }
template<unsigned id, unsigned... ids>
std::ostream & operator << (std::ostream& out, Index_Set<id, ids...>) {
    return out << id << ' ' << Index_Set<ids...>();
}

void test_multiplication(int nw = 1){
    set_workers_number(nw);
    auto start = std::chrono::steady_clock::now();
    tensor<int> t3(3,2,3), t4(2);

    for(auto iter=t3.begin(); iter!=t3.end(); ++iter)
        *iter = 1;
    for(auto iter=t4.begin(); iter!=t4.end(); ++iter)
        *iter = 1;

    auto i=new_index;
    auto j=new_index;
    auto k=new_index;

    tensor<int,rank<2>> t6=t3(i,j,k)*t4(j);
    std::chrono::duration<double> time = std::chrono::steady_clock::now() - start;
    std::cout << "Elapsed time: " << std::chrono::duration<double, std::milli>(time).count() << " ms.\n";
    std::cout << std::endl;
}

void test_long_multiplication(int nw = 1){
    set_workers_number(nw);
    auto start = std::chrono::steady_clock::now();
    tensor<int,rank<2>> t1(2,3), t2(3,2);
    tensor<int> t6(2);

    for(auto iter=t1.begin(); iter!=t1.end(); ++iter)
        *iter = 1;
    for(auto iter=t2.begin(); iter!=t2.end(); ++iter)
        *iter = 1;

    auto i=new_index;
    auto j=new_index;
    auto k=new_index;

    t6(k)=t1(k,j)*t1(i,i)*t2(j,j)*t2(j,j)*t1(i,i)*t2(j,j)*t2(j,j)*t1(i,i)*t2(j,j)*t2(j,j)*t1(i,i);
    std::chrono::duration<double> time = std::chrono::steady_clock::now() - start;
    std::cout << "Elapsed time: " << std::chrono::duration<double, std::milli>(time).count() << " ms.\n";
    std::cout << std::endl;
}

void test_sum(int nw = 1){
    set_workers_number(nw);
    auto start = std::chrono::steady_clock::now();
    tensor<int> t1(2,2), t3(2,2,3,3), t4(2,3,3,2);

    for(auto iter=t3.begin(); iter!=t3.end(); ++iter)
        *iter = 1;
    for(auto iter=t4.begin(); iter!=t4.end(); ++iter)
        *iter = 1;

    auto i=new_index;
    auto j=new_index;
    auto k=new_index;

    t1(i,j) = t3(i,j,k,k)+t4(i,k,k,j);
    std::chrono::duration<double> time = std::chrono::steady_clock::now() - start;
    std::cout << "Elapsed time: " << std::chrono::duration<double, std::milli>(time).count() << " ms.\n";
    std::cout << std::endl;
}

void test_big_sum(int nw = 1){
    set_workers_number(nw);
    auto start = std::chrono::steady_clock::now();
    tensor<int,rank<5>> t1(2,2,8,200,100);
    tensor<int,rank<4>> t3(2,2,8,9);

    for(auto iter=t1.begin(); iter!=t1.end(); ++iter)
        *iter = 1;
    for(auto iter=t3.begin(); iter!=t3.end(); ++iter)
        *iter = 1;

    auto i=new_index;
    auto j=new_index;
    auto k=new_index;

    tensor<int> t4 = t3(i,j,k,k)+t1(i,j,k,k,k)+t3(i,j,k,k)+t1(i,j,k,k,k)+t3(i,j,k,k)+t1(i,j,k,k,k)+t3(i,j,k,k)+t1(i,j,k,k,k)+t3(i,j,k,k)+t1(i,j,k,k,k)+t3(i,j,k,k)+t1(i,j,k,k,k)+t3(i,j,k,k)+t1(i,j,k,k,k)+t3(i,j,k,k)+t1(i,j,k,k,k);
    std::chrono::duration<double> time = std::chrono::steady_clock::now() - start;
    std::cout << "Elapsed time: " << std::chrono::duration<double, std::milli>(time).count() << " ms.\n";
    std::cout << std::endl;
}

void test_sum_w(){
    // sum
    std::cout << "Sum test: " << std::endl;
    test_sum(1);
    sleep(1);
    test_sum(2);
    sleep(1);
    test_sum(4);
    sleep(1);
    test_sum(8);
    sleep(1);
    std::cout << "________________________________ " << std::endl;
    std::cout << "Big sum test: " << std::endl;
    test_big_sum(1);
    sleep(1);
    test_big_sum(2);
    sleep(1);
    test_big_sum(4);
    sleep(1);
    test_big_sum(8);
    sleep(1);
    std::cout << "________________________________ " << std::endl;
}

void test_mult_w(){
    // multiplication
    std::cout << "Multiplication test: " << std::endl;
    test_multiplication(1);
    sleep(1);
    test_multiplication(2);
    sleep(1);
    test_multiplication(4);
    sleep(1);
    test_multiplication(8);
    sleep(1);
    std::cout << "________________________________ " << std::endl;
    std::cout << "Long Multiplication test: " << std::endl;
    test_long_multiplication(1);
    sleep(1);
    test_long_multiplication(2);
    sleep(1);
    test_long_multiplication(4);
    sleep(1);
    test_long_multiplication(8);
    sleep(1);
}

void test_invert(int nw = 1){
    set_workers_number(nw);
    auto start = std::chrono::steady_clock::now();
    tensor<int,rank<3>> t1(4,40,3), t2(3,40,4);
    int val = 0;
    int pcount = 0;
    for(auto iter=t1.begin(); iter!=t1.end(); ++iter)
        *iter = val++;
    std::cout << '\n';
    auto i=new_index;
    auto j=new_index;
    auto k=new_index;

    t2(k,j,i) = t1(i,j,k);
    pcount = 0;

    std::chrono::duration<double> time = std::chrono::steady_clock::now() - start;
    std::cout << "Elapsed time: " << std::chrono::duration<double, std::milli>(time).count() << " ms.\n";
    std::cout << std::endl;

}

void test_inv_w(){
    // inversion
    std::cout << "Inversion test: " << std::endl;
    test_invert(1);
    sleep(1);
    test_invert(2);
    sleep(1);
    test_invert(4);
    sleep(1);
    test_invert(5);
    sleep(1);
    test_invert(8);
    std::cout << "________________________________ " << std::endl;
}

int main(){
    test_inv_w();
    test_sum_w();
    test_mult_w();
    return 0;
}