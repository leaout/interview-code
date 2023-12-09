#include <iostream>
#include <unordered_map>
#include <chrono>
#include <thread>
#include <vector>
#include <mutex>
#include <list>
#include <set>
#include <unordered_set>
#include <map>

using namespace std;
/*
 * 设计一个系统，统计url访问耗时，访问次数。
 * */
std::mutex g_mtx;
unordered_map<string,list<pair<time_t,int>>> g_map;
void add(const string& url, time_t ts, int count){
    std::lock_guard<mutex> lock(g_mtx);
    g_map[url].emplace_back(std::make_pair(ts,count));
}

int query(const string& url, time_t begin_ts, time_t end_ts){
    std::lock_guard<mutex> lock(g_mtx);
    auto find_ret = g_map.find(url);
    int ret = 0;
    if(find_ret == g_map.end()){
        return ret;
    }

    for(auto& pr : find_ret->second){
        if(pr.first >= begin_ts && pr.first <= end_ts){
            ret += pr.second;
        }
    }
    return ret;
}

/*
 * binary search
 * */
int bid_search(const int num[],int left,int right, int target) {
    while(left <= right){
        int mid = left+(right-left)/2;
        if(num[mid] == target){
            return mid;
        }else if(num[mid] < target){
            left = mid+1;
        } else{
            right = mid -1;
        }
    }
    return -1;
}

/*
 *判断一个链表是否有环，找出环的起点
 * */
struct Node{
    int val;
    Node* next = nullptr;
};
bool check_cir(Node* head){
    Node* slow = head;
    Node* fast = head;
    while(slow && fast){
        slow = slow->next;
        fast = fast->next;
        if(fast != nullptr){
            fast = fast->next;
        }
        if(slow == fast){
            return true;
        }
    }
    return false;
}

Node* find_circle_entry(Node* head){
    unordered_set<Node*> marks;
    Node* tmp = head;
    Node* pre = nullptr;
    while(tmp){
        auto ret = marks.insert(tmp);
        if(ret.second == false){
            return pre;
        }
        pre = tmp;
        tmp = tmp->next;
    }
    return nullptr;
}
/*
 * 输入一个10进制数字，输出这个数8进制表示
    例子：
    输入： 10
    输出： 12
    测试输入: 1000000
 *
 * */
int dec2oct(int dec){
    int ret = 0;
    int i = 1;
    while (dec != 0){
        ret += (dec %8) *i;
        dec /=8;
        i*=10;
    }
    return ret;
}
void dec2oct_test(){
    std::cout << "10to dec:" <<  dec2oct(10) << std::endl;
    std::cout << "1000000 to dec:" <<  dec2oct(1000000) << std::endl;
}


/*
 * 输入一个无向图邻接矩阵A（Aij=1代表i点和j点相连，0代表不相连）。输出这个图的联通分量的个数
 * （一个联通分量就是一个子图，该子图每两个点间都可以有路径到达）。输入第一行是这个图点的个数。
    例子
    输入：
    5
    0 1 1 0 0
    1 0 1 0 0
    1 1 0 0 0
    0 0 0 0 1
    0 0 0 1 0
    输出
    2
    （注：该图有两个联通分量，一个是{1,2,3}， 一个是{4,5}）

    测试输入：
    7
    0 1 0 0 0 1 0
    1 0 0 0 0 0 0
    0 0 0 1 1 0 0
    0 0 1 0 1 0 0
    0 0 1 1 0 0 0
 *
 * */
void find_connected(int node_num,vector<vector<int>> matr){
    vector<set<int>> conn;
    std::set<int> accessed;
    for(int i = 0; i < node_num; ++i){
        auto ret = accessed.insert(i);
        if(ret.second){
            set<int> tmp_nodes;
            vector<int> current_nodes;
            vector<int> next_nodes;
            current_nodes.emplace_back(i);
            tmp_nodes.insert(i);
            while(!current_nodes.empty()){
                for(auto&v : current_nodes){
                    for(int j = 0; j < node_num; ++j){
                        if(accessed.count(j) > 0){
                            continue;
                        }

                        if(matr[v][j] == 1){
                            accessed.emplace(j);
                            next_nodes.emplace_back(j);
                            tmp_nodes.insert(j);
                        }
                    }
                }
                current_nodes.swap(next_nodes);
                next_nodes.clear();
            }
            conn.emplace_back(tmp_nodes);
        }
    }
    std::cout << "connecte size: " << conn.size() << std::endl;
    for(auto& set_val : conn){
        for(auto&val : set_val){
            std::cout  << val+1 << ",";
        }
        std::cout << std::endl;
    }
}
void test_find_connected_size(){
//    vector<vector<int>> matr = {{1, 4, 3},
//                                {2, 3, 1},
//                                {2, 3, 4}};
//    vector<vector<int>> matr = {{0, 1, 1, 0, 0},
//                                {1, 0, 1, 0, 0},
//                                {1, 1, 0, 0, 0},
//                                {0, 0, 0, 0, 1},
//                                {0, 0, 0, 1, 0}};
    vector<vector<int>> matr = {{0, 1, 0, 0, 0, 1, 0},
                                {1, 0, 0, 0, 0, 0, 0},
                                {0, 0, 0, 1, 1, 0, 0},
                                {0, 0, 1, 0, 1, 0, 0},
                                {0, 0, 1, 1, 0, 0, 0},
                                {1, 0, 0, 0, 0, 0, 1},
                                {0, 0, 0, 0, 0, 1, 0}};
    find_connected(7, matr);
}
/*
 * 给定一个矩阵，找出从左上到右下角的一条路，使得这条路上数字和最大。这条路前进的方向只能向右或向下。
 * 输入的第一行是矩阵的行数和列数。输出第一行是一个序列，为该条路上的数字，第二行是这些数字的和。
    例子
    输入：
    3 3
    1 4 3
    2 3 1
    2 3 4
    输出
    1 4 3 3 4
    15

    测试输入：
    5 5
    1 1 1 1 2
    2 3 4 1 4
    3 1 4 2 4
    2 1 5 7 2
    4 3 3 4 5
 * */

class SolMaxPath{
public:
    int m_row_max;
    int m_col_max;
    vector<vector<int>>* m_matrix;

    void find_max_path(vector<vector<int>> &matrix,int row_size,int col_size) {
        m_row_max = row_size-1;
        m_col_max = col_size-1;
        m_matrix = &matrix;
        map<int,vector<int>> path;
        pair<int,vector<int>> tmp_path;
        map<int,vector<int>> result;
        tmp_path.first += matrix[0][0];
        tmp_path.second.emplace_back(matrix[0][0]);
        int i = 0,j = 0;
        dfs_path(i,j,tmp_path,result);

        auto res = result.rbegin();

        std::cout << "max : "<<res->first << std::endl;
       for(auto&val : res->second){
           std::cout << val << ",";
       }
       std::cout << std::endl;
    }

    void dfs_path(int& i, int& j,pair<int,vector<int>>& path, map<int,vector<int>> &result){
        if(i == m_row_max && j == m_col_max){
            result.insert(path);
            return;
        }
        for(int direct = 0; direct <2; direct++){
            if(direct){
                if(i < m_row_max){
                    ++i;
                    if(i <= m_row_max){

                        path.first += (*m_matrix)[i][j];
                        path.second.emplace_back((*m_matrix)[i][j]);
                        dfs_path(i, j, path, result);
                        --i;
                        path.first -= path.second.back();
                        path.second.pop_back();
                    }
                }
            } else{
                if(j < m_col_max){
                    ++j;
                    if(j <= m_col_max ){
                        path.first += (*m_matrix)[i][j];
                        path.second.emplace_back((*m_matrix)[i][j]);
                        dfs_path(i,j,path,result);
                        --j;
                        path.first -= path.second.back();
                        path.second.pop_back();
                    }
                }
            }
        }
        return ;
    }
};

void test_find_max_path() {
    vector<vector<int>> matr = {{1, 1, 1, 1, 2},
                                {2, 3, 4, 1, 4},
                                {3, 1, 4, 2, 4},
                                {2, 1, 5, 7, 2},
                                {4, 3, 3, 4, 5}};
    SolMaxPath s1;
    s1.find_max_path(matr, 5, 5);

}
/*
 * 给出一个字符串 S，考虑其所有重复子串（S 的连续子串，出现两次或多次，可能会有重叠）。
    返回任何具有最长可能长度的重复子串。（如果 S 不含重复子串，那么答案为 ""。）
 * */
class SolLonggestSubString {
public:
    //检查是针对存在重复子串 还是发生了哈希冲突
    bool check_hash(string& s, pair<int, int>& a, pair<int, int> b) {       //查看是否是真重复子串还是因为发生哈希碰撞而导致哈希值相同
        for (int i = a.first, j = b.first; i != a.second && j != b.second; ++i, ++j) {
            if (s[i] != s[j]) return false;
        }
        return true;
    }
    //检查字符串s中是否存在长度为len的重复子串，如果有则返回该子串，否则返回空字符串
    string check(string& s, int len){
        int base = 26;//二十六个字母对应二十六进制
        int mod = 1000007;//取模 避免哈希冲突

        int num = 0;
        for(int i = 0; i < len; i++)//计算出第一个len长度的哈希映射值
            num = (num * base + s[i] - 'a')%mod;

        unordered_map<int, pair<int, int>> seen;//存储的是哈希映射值及对应的坐标
        seen.insert({num, {0, len - 1}});

        int al = 1;//根据公式 计算出常数a的len次方
        for(int i = 1; i <= len; i++)
            al = (al * base)%mod;

        //遍历字符串 计算每一个长度为len的子串的哈希映射值
        for(int i = 1; i < s.size() - len + 1; i++){

            //计算长度为len的子串的哈希映射值
            num = num * base - ((s[i-1] - 'a') * al)%mod;
            while(num < 0) num += mod;
            num = (num + (s[i + len - 1] - 'a'))%mod;

            //查找是否有重复的子串
            if(seen.count(num))
                if(check_hash(s, seen[num], {i, i + len - 1}))
                    return s.substr(i, len);//如果是真的存在而不是因为哈希冲突，就返回这个子串
            seen.insert({num, {i, i + len - 1}});//如果是哈希冲突 就插入
        }
        return "";
    }

    //返回字符串s最长重复子串
    string longestDupSubstring(string s) {
        int m = s.size();
        int left = 0, right = m;
        string res = "";
        while(left < right){
            int mid = left + (right - left)/2;//二分法找到最长重复子串
            string tmp = check(s, mid);
            if(!tmp.empty()){//如果存在重复子串，就保存下来最长的一个重复子串
                res = tmp.size() > res.size() ? tmp : res;
                left = mid + 1;
            }else
                right = mid;

        }
        return res;
    }
};
/*给你一个整数 x ，如果 x 是一个回文整数，返回 true ；否则，返回 false 。

回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。

例如，121 是回文，而 123 不是。
 */
bool isPalindrome(int x) {
    int tmp = x;
    if(x < 0){
        return false;
    }
    long revers_x = 0;
    while (x) {
        auto v = x % 10;
        if (x) {
            revers_x +=v;
        }
        x /= 10;
        if (x) {
            revers_x*=10;
        }
    }
    return (tmp == revers_x);
}

/*
 * 一个自重复串是一个字符串，其前一半和后一半是一样的，例如 abcdbabcdb （长度一定是偶数）。
    输入一个字符串，找出其中最长的自重复子串。这里的子串要求连续。

    例子
    输入：
    abababcdabcd
    输出：
    abcdabcd


    测试输入：faaacabcddcbabcddcbedfgaac
 *
 * */

/*
 * 有两个数据集合如下
    A {98,38,48,17,21,27,456,9887,100,2358}
    B {8929,74,1994,12,485,537,183,134,745}
    题目一
  1 分别对集合 A B 做排序，排序为从小到大
  2 把排序后的集合 A B 分别存入到两个链表ListA ListB 中
  3 把 排序后的ListA 和 排序后的ListB 再合并为一个有序的链表 ListC，输出ListC中所有的数据


要求：
  用C语言，其中排序、链表，不使用函数库
  输出结果请截图一起发出
 * */
void swap(int* a, int* b) {
    if(*a != *b){
        *a ^= *b;
        *b ^= *a;
        *a ^= *b;
    }
}

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);
    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

void quickSort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}
struct ListNode{
    int val;
    ListNode* next;
};
ListNode* arr2list(int arr[], int len){
    ListNode* head = nullptr;
    ListNode* current = nullptr;

    for(int i = 0; i < len; ++i){
        if(i == 0){
            head = new ListNode{arr[i], nullptr};
            current = head;
        } else{
            current->next = new ListNode{arr[i], nullptr};
            current = current->next;
        }
    }
    return head;
}
ListNode* merge_list(ListNode* A, ListNode* B){
    ListNode head ;
    ListNode* current = &head;
    while(A || B){
        if(A && B){
            if(A->val <= B->val){
                current->next = new ListNode{A->val, nullptr};
                current = current->next;
                A = A->next;
            } else{
                current->next = new ListNode{B->val, nullptr};
                current = current->next;
                B = B->next;
            }
        } else if (A && (B == nullptr)) {
            current->next = new ListNode{A->val, nullptr};
            current = current->next;
            A = A->next;
        } else{
            current->next = new ListNode{B->val, nullptr};
            current = current->next;
            B = B->next;
        }
    }
    return head.next;
}
void test_1(){
    int arr1[] = {98,38,48,17,21,27,456,9887,100,2358};
    int arr2[] = {8929,74,1994,12,485,537,183,134,745};
    int arr1_len = sizeof(arr1)/4 ;
    int arr2_len = sizeof(arr2)/4 ;
    quickSort(arr1,0, arr1_len-1);
    quickSort(arr2,0, arr2_len-1);

    ListNode* listA = arr2list(arr1,arr1_len);
    ListNode* listB = arr2list(arr2,arr2_len);
    ListNode* ListC = merge_list(listA,listB);
    ListNode *current = ListC;
    while (current) {
        printf("%d ", current->val);
        current = current->next;
    }
}

/*
 *题目二  双链表 实现增 删 功能

A {98,38,48,16,20,27,456,9887,100,2358}

添加到双链表 成功后 再删除 节点9887

要求：
  用C语言，其中排序、链表，不使用函数库
  输出结果请截图一起发出
 * */
struct ListNodeD{
    int val;
    ListNodeD* pre = nullptr;
    ListNodeD* next = nullptr;
};
ListNodeD* append(ListNodeD*tail, int val){
    tail->next = new ListNodeD{val, tail, nullptr};
    return tail->next;
}
ListNodeD* init(){

    int arr[] = {98, 38, 48, 16, 20, 27, 456, 9887, 100, 2358};
    int len = sizeof(arr) / 4;

    ListNodeD head ;
    ListNodeD* current = &head;
    for (int i = 0; i < len; ++i) {
        current->next = new ListNodeD{arr[i], current,nullptr};
        current = current->next;
    }
    return head.next;
}
void remove(ListNodeD*head, int val){
    while(head){
        if(head->val == val){
            head->pre->next = head->next;
            head->next->pre = head->pre;
            return;
        } else{
            head = head->next;
        }
    }
}

void print(ListNodeD *head) {
    while (head) {
        printf("%d ", head->val);
        head = head->next;
    }
    printf("\n");
}
void test_2(){
    ListNodeD* list = init();
    print(list);
    remove(list,9887);
    print(list);
}


int main() {
    return 0;
}

