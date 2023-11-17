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

int main() {
    return 0;
}

