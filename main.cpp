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
int main() {
    SolLonggestSubString sls;
    std ::cout << "result:" << sls.longestDupSubstring("faaacabcddcbabcddcbedfgaac") << std::endl;
    return 0;
}

