[TOC]
# 算法思想

## 1. 贪心

[406. Queue Reconstruction by Height](https://leetcode.com/problems/queue-reconstruction-by-height/)

### 1.按身高和序号重组队列

题目描述：一个学生用一个数组表示[h,k]h代表学生的高度，k代表站在学生前面比他高或者一样高的学生的人数

思路：身高降序排序，k升序排序，将某个学生插入第k个位置，从身高高的开始插，不影响后续操作，如果从身高小的开始，再插入身高大的之后，原来的k会变成k+1

```java
public int[][] reconstructQueue(int[][] people) {
    Arrays.sort(people, (a,b)->{
       return (a[0] == b[0]) ? a[1] - b[1] : b[0] - a[0];
    });

    List<int[]> result = new LinkedList<>(); //插入多，使用linkedlist

    for(int[] a : people){
        result.add(a[1],a);
    }

    return result.toArray(people);
}
```

### 2.买卖股票的最大收益

[121. Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)

题目描述：进行一次交易包含一次买和卖，得到最大的收益

思路：不断更新最小的买入价格，用当前的价格做为卖出价格更新最大利润

```java
public int maxProfit(int[] prices) {
    int result = 0, buy = Integer.MAX_VALUE;

    for(int i = 0; i < prices.length; i++) {

        buy = Math.min(buy, prices[i]);

        result = Math.max(result, prices[i] - buy);

    }

   return result;
}
```

### 3.分配饼干

[455. Assign Cookies](https://leetcode.com/problems/assign-cookies/)

```java
public int findContentChildren(int[] g, int[] s) {
    TreeMap<Integer, Integer> map = new TreeMap<>();

    for(int num : g)
        map.put(num, map.getOrDefault(num, 0) + 1);

    int res = 0;
    for(int num : s){
        Integer key = map.floorKey(num);
        if(key != null){
            int temp = map.get(key) - 1;
            res++;
            if(temp == 0){
                map.remove(key);
            }else{
                map.put(key, temp);
            }
        }
    }

    return res;

}
```

### 4.判断是否为子序列

[392. Is Subsequence](https://leetcode.com/problems/is-subsequence/)

```java
public boolean isSubsequence(String s, String t) {

    int lens = s.length(), lent = t.length();

    int i = 0, j = 0;
    while( i < lens && j < lent){
        if(s.charAt(i) == t.charAt(j++)){
            i++;
        }

    }

    return i == lens;
}
```

### 5.不重叠区间的个数

[435. Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/)

动态规划，类似最长递增子序列

```java
public int eraseOverlapIntervals(int[][] intervals) {

    Arrays.sort(intervals,(a,b) -> {
        return a[0] - b[0] == 0 ? a[1] - b[1] : a[0] - b[0];
    });

    int n = intervals.length;
    int[] dp = new int[n];

    int res = 0;
    for (int i = 0; i < n; i++){
        dp[i] = 1;

        for (int j = 0; j < i; j++){
            if (intervals[i][0] >= intervals[j][1]){
                dp[i] = Math.max(dp[i], dp[j] + 1);
            }
        }
        res = Math.max(res,dp[i]);
    }


    return n - res;
}
```

贪心思想

```java
public int eraseOverlapIntervals(int[][] intervals) {

    Arrays.sort(intervals, (a, b) -> a[1] - b[1]);

    int n = intervals.length, flag = 0, res = 0;

    for(int i = 1; i < n; i++){
        if(intervals[i][0] < intervals[flag][1]){
            res++;
        }else {
            flag = i;
        }
    }

    return res;

}
```

## 2. 搜索

###  Backtracking

Backtracking（回溯）属于 DFS。

- 普通 DFS 主要用在  **可达性问题** ，这种问题只需要执行到特点的位置然后返回即可。
- 而 Backtracking 主要用于求解  **排列组合** 问题，例如有 { 'a','b','c' } 三个字符，求解所有由这三个字符排列得到的字符串，这种问题在执行到特定的位置返回之后还会继续执行求解过程。

因为 Backtracking 不是立即返回，而要继续求解，因此在程序实现时，需要注意对元素的标记问题：

- 在访问一个新元素进入新的递归调用时，需要将新元素标记为已经访问，这样才能在继续递归调用时不用重复访问该元素；
- 但是在递归返回时，需要将元素标记为未访问，因为只需要保证在一个递归链中不同时访问一个元素，可以访问已经访问过但是不在当前递归链中的元素。

#### 1. 排列

[46. Permutations](https://leetcode.com/problems/permutations/)

```java
public List<List<Integer>> permute(int[] nums) {
    List<List<Integer>> ret = new ArrayList<>(); //结果集
    List<Integer> temp = new ArrayList<>();   //保存排列
    boolean[] hasVisited = new boolean[nums.length]; //访问标志
    backTracking(ret, temp, hasVisited, nums);
    return ret;
}

private void backTracking(List<List<Integer>> ret,
                          List<Integer> temp,  
                          boolean[] hasVisited, int[] nums){
    int len = nums.length;
    if(temp.size() == len){
        ret.add(new ArrayList<>(temp));
        return;
    }

    for(int i = 0; i < len; i++){

        if(hasVisited[i]){
            continue;
        }

        hasVisited[i] = true; //标记为已访问
        temp.add(nums[i]);
        backTracking(ret, temp, hasVisited, nums);
        temp.remove(temp.size() - 1); //下一个排列
        hasVisited[i] = false; //递归结束，元素变为可访问
    }

}
```

#### 2. 子集

[78. Subsets](https://leetcode.com/problems/subsets/)

找出集合的所有子集

```java
public List<List<Integer>> subsets(int[] nums) {

    List<List<Integer>> res = new ArrayList<>();

    for(int i = 0; i <= nums.length; i++)
        backTrack(nums, 0, i, new ArrayList<>(), res);
    return res;

}

private void backTrack(int[] nums, int start, int k, List<Integer> temp, List<List<Integer>> res){
    if(temp.size() == k){
        res.add(new ArrayList<>(temp));
        return;
    }

  	//剪枝，所剩元素小于所需要的元素时，返回
    while(nums.length - start + 1 > k - temp.size()){
        temp.add(nums[start++]);
        backTrack(nums, start, k, temp, res);
        temp.remove(temp.size() - 1);
    }
}
```

#### 3. 组合求和

[39. Combination Sum](https://leetcode.com/problems/combination-sum/)

```java
public List<List<Integer>> combinationSum(int[] candidates, int target) {

    List<List<Integer>> result = new ArrayList<>();
    List<Integer> temp = new ArrayList<>();

    backTracking(result, temp, target, 0, candidates);
    return result;
}

private void backTracking(List<List<Integer>> result,
                          List<Integer> temp, 
                          int target, int start,
                          int[] candidates){
    if(target == 0){
        result.add(new ArrayList(temp));
      
    }

  	//防止添加之前的元素
    for(int i = start; i < candidates.length; i++){
        if(candidates[i] <= target){
            temp.add(candidates[i]);
            backTracking(result, temp, target - candidates[i],
                         i, candidates);
            temp.remove(temp.size() - 1);
        }
    }

}
```

#### 4. 数字键盘组合

[17. Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/)

```java
private String[] keys = {" ","","abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"};
public List<String> letterCombinations(String digits) {
    List<String> res = new ArrayList<>();

    if(digits == null || digits.length() == 0){
        return res;
    }

    doCombinations(res, digits, new StringBuilder());
    return res;

}

public void doCombinations(List<String> res, String digits, StringBuilder prefix){

    if(prefix.length() == digits.length()){
        res.add(prefix.toString());
        return;
    }

    int index = digits.charAt(prefix.length()) - '0';

    String temp = keys[index];

    for(char c : temp.toCharArray()){
        prefix.append(c);
        doCombinations(res, digits, prefix);
        prefix.deleteCharAt(prefix.length() - 1);
    }
}
```

#### 5. IP地址的划分

[93. Restore IP Addresses](https://leetcode.com/problems/restore-ip-addresses/)

```java
public List<String> restoreIpAddresses(String s) {

    List<String> result = new ArrayList<>();

    doRestore(0, new StringBuilder(), result, s);

    return result;


}

private void doRestore(int k, StringBuilder tempAddress, List<String> address, String s) {
    if(k == 4 || s.length() == 0){
        if(k == 4 && s.length() == 0)
            address.add(tempAddress.toString());
        return;
    }


    for(int i = 0; i < s.length() && i <= 2; i++) {
        if(i != 0 && s.charAt(0) == '0' ) //防止出现025.这种情况
            break;

        String part = s.substring(0, i + 1);

        if(Integer.valueOf(part) <= 255){

            if(tempAddress.length() != 0){
               part = "." + part;
            }

            tempAddress.append(part);


            doRestore(k + 1, tempAddress, address, s.substring(i + 1));
            tempAddress.delete(tempAddress.length() - part.length(), tempAddress.length());
        }
    }
}
```

#### 6. 分割字符串使得每个部分都是回文数

[131. Palindrome Partitioning](https://leetcode.com/problems/palindrome-partitioning/)

```java
public List<List<String>> partition(String s) {

    List<List<String>> res = new ArrayList<>();

    doPartition(s, res, new ArrayList<>());

    return res;
}


private void doPartition(String s, List<List<String>> partitions, List<String> tempPartition){

    if(s.length() == 0){
        partitions.add(new ArrayList<>(tempPartition));
        return;
    }

    for(int i = 0; i < s.length(); i++){
        if(isPalindrome(s, 0, i)){
            tempPartition.add(s.substring(0, i + 1));
            doPartition(s.substring(i + 1), partitions, tempPartition);
            tempPartition.remove(tempPartition.size() - 1);
        }

    }
}

private boolean isPalindrome(String s, int start, int end) {
    while(start < end){
        if(s.charAt(start++) != s.charAt(end--))
            return false;
    }

    return true;
}
```

#### 7. 含有相同元素求排列

[47. Permutations II](https://leetcode.com/problems/permutations-ii/)

```java
public List<List<Integer>> permuteUnique(int[] nums) {

    List<List<Integer>> res = new ArrayList<>();
    Arrays.sort(nums); //先排序
    boolean[] isVisited = new boolean[nums.length];
    backTrack(isVisited, res, new ArrayList<>(), nums);
    return res;

}

private void backTrack(boolean[] isVisited, List<List<Integer>> res, List<Integer> temp, int[] nums){

    if(temp.size() == nums.length){
        res.add(new ArrayList<>(temp));
        return;
    }


    for(int i = 0; i < nums.length; i++){
        if(isVisited[i]){
            continue;
        }

      	//剪枝，防止添加重复元素
        if(i != 0 && nums[i] == nums[i - 1] && !isVisited[i - 1]){ 
            continue;
        }

        temp.add(nums[i]);
        isVisited[i] = true;
        backTrack(isVisited, res, temp, nums);
        isVisited[i] = false;
        temp.remove(temp.size() - 1);
    }        
}
```

#### 8. 组合

[77. Combinations](https://leetcode.com/problems/combinations/)

```java
public List<List<Integer>> combine(int n, int k) {
    List<List<Integer>> res = new ArrayList<>();
    backTrack(1, res, new ArrayList<>(), k, n);
    return res;

}

private void backTrack(int start, List<List<Integer>> res, List<Integer> temp, int k, int n){
    if(k == 0){
        res.add(new ArrayList<>(temp));
        return;
    }

  	//剪枝，当所剩元素少于所需要的元素时，直接返回
    while(start <= n - k + 1){
        temp.add(start++);
        backTrack(start, res, temp, k - 1, n);
        temp.remove(temp.size() - 1);
    }

}
```

#### 9. 含有相同元素的组合求和

[40. Combination Sum II](https://leetcode.com/problems/combination-sum-ii/)

```java
public List<List<Integer>> combinationSum2(int[] candidates, int target) {

    List<List<Integer>> res = new ArrayList<>();
  	//排序
    Arrays.sort(candidates);
    backTrack(0, target, res, new ArrayList<>(), candidates);
    return res;

}

private void backTrack(int start, int target, List<List<Integer>> res, List<Integer> temp, int[] candidates) {
    if(target == 0){
        res.add(new ArrayList<>(temp));
        return;
    }

    while(start < candidates.length) {

        if(candidates[start] <= target) {

            temp.add(candidates[start]);
            backTrack(start + 1, target - candidates[start], res, temp, candidates);
            temp.remove(temp.size() - 1);

          	//剪枝，跳过相同元素
            while(start < candidates.length - 1 && candidates[start] == candidates[start + 1]){
                start++;
            }

        }
        start++;
    }
}
```

#### 10. 1-9数字的组合求和

[216. Combination Sum III](https://leetcode.com/problems/combination-sum-iii/)

```java
public List<List<Integer>> combinationSum3(int k, int n) {

    List<List<Integer>> res = new ArrayList<>();
    backTrack(res, k, n, 1, new ArrayList<>());
    return res;

}

private void backTrack(List<List<Integer>> res, int k, int n, int start, List<Integer> temp) {
    if(k == 0 || n <= 0){
        if(n == 0 && k == 0)
            res.add(new ArrayList<>(temp));
        return;
    }

    while(start <= 9){
        if(start <= n){
            temp.add(start);
            backTrack(res, k-1, n - start, start + 1, temp);
            temp.remove(temp.size() - 1);
        }
        start++;
    }
}
```

#### 11. 含有相同元素求子集

[90. Subsets II](https://leetcode.com/problems/subsets-ii/)

```java
public List<List<Integer>> subsetsWithDup(int[] nums) {

  	//排序
    Arrays.sort(nums);

    List<List<Integer>> res = new ArrayList<>();

    for(int i = 0; i <= nums.length; i++)
        backTrack(nums, 0, res, new ArrayList<>(), i);
    return res;

}

private void backTrack(int[] nums, int start, List<List<Integer>> res, List<Integer> temp, int size) {

    if(temp.size() == size){
        res.add(new ArrayList<>(temp));
        return;
    }

  	//剪枝
    while(nums.length - start + 1 > size - temp.size()){

        temp.add(nums[start]);
        backTrack(nums, start + 1, res, temp, size);
        temp.remove(temp.size() - 1);
      	//跳过重复元素
        while(start < nums.length - 1 && nums[start] == nums[start + 1])
            start++;
        start++;
    }

}
```

#### 12. 在矩阵中寻找字符串

[79. Word Search](https://leetcode.com/problems/word-search/)

```java
class Solution {
    
    private int m;
    private int n;
    private int[][] d ={{0,-1},{0,1},{-1,0},{1,0}};
    
    public boolean exist(char[][] board, String word) {
        
       
        m = board.length;
        n = board[0].length;
        
        boolean[][] isVisited = new boolean[m][n];
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                if(backTrack(board, word, 0, i, j, isVisited)){
                    return true;
                }
            }
        }
        
        return false;
        
    }
    
    private boolean backTrack(char[][] board, String word, int index, int x, int y, boolean[][] isVisited){
        
        if(index == word.length() - 1){
            return board[x][y] == word.charAt(index);
        }

        if(board[x][y] == word.charAt(index)){
            
            isVisited[x][y] = true;
            for(int i = 0; i < 4; i++){
                int newX = x + d[i][0];
                int newY = y + d[i][1];
                if(inArea(newX, newY) && !isVisited[newX][newY] && board[x][y] == word.charAt(index)){
                   
                    if(backTrack(board, word, index + 1, newX, newY, isVisited)){
                        return true;
                    }    
                }
            }
            isVisited[x][y] = false;
        }
        
        return false; 
    }
    
    private boolean inArea(int x, int y){
        return x >= 0 && x < m && y >= 0 && y < n;
    }
    
    
}
```

#### 13. N皇后

[51. N-Queens](https://leetcode.com/problems/n-queens/)

判断是否在45的对角线，总共有2*n - 1根对角线，(x,y),x+y等于定值

判断是否在135对角线，总共有2*n - 1根对角线，(x,y),x-y等于定值

```java
class Solution {
    
    private boolean[] col;
    private boolean[] dia1;
    private boolean[] dia2;
    public List<List<String>> solveNQueens(int n) {
        
        col = new boolean[n];
        dia1 = new boolean[2 * n - 1];
        dia2 = new boolean[2 * n - 1];
        
        List<List<String>> res = new ArrayList<>();
        backTrack(res, new ArrayList<>(), n, 0);
        
        return res;
    }
    
    private void backTrack(List<List<String>> res, List<String> temp, int n, int nowC){

        if(temp.size() == n){
            res.add(new ArrayList<>(temp));
            return;
        }
        
        for(int i = 0; i < n; i++){ 
            int d1 = i + nowC, d2 = i - nowC + n - 1;
            if(!col[i] && !dia1[d1] && !dia2[d2]){
                temp.add(genStr(i, n));
                col[i] = true;
                dia1[d1] = true;
                dia2[d2] = true;
                
                backTrack(res, temp, n, nowC + 1);
                temp.remove(temp.size() - 1);
                col[i] = false;
                dia1[d1] = false;
                dia2[d2] = false;
            }            
            
        }
        
    }
    
    
    private String genStr(int index, int n){
        StringBuilder builder = new StringBuilder();
        for(int i = 0; i < n; i++){
           if(i == index){
               builder.append('Q');
           }else {
               builder.append('.');
           }
        }
        return builder.toString();       
    }
}
```

#### 14. 数独

[37. Sudoku Solver](https://leetcode.com/problems/sudoku-solver/)

```java
class Solution {
    
    
    boolean[][] col = new boolean[9][10];
    boolean[][] row = new boolean[9][10];
    boolean[][] sub = new boolean[9][10];
    boolean[][] isVisited = new boolean[9][9];
    
    
    public void solveSudoku(char[][] board) {
        
        for(int i = 0; i < 9; i++){
            for(int j = 0; j < 9; j++){
                if(board[i][j] != '.'){
                    int val = board[i][j] - '0';
                    col[j][val] = true;
                    row[i][val] = true;
                    sub[calNum(i, j)][val] = true;
                    isVisited[i][j] = true;
                }
            }
        }
        
       
        trackBack(board, 0, 0);
        
        
    }
    
    private boolean trackBack(char[][] board, int x, int y){
        
      	//递归到最后，已经找到解
        if(x >= 9){
            return true ;
        }
        
        
        if(!isVisited[x][y]){
            
            isVisited[x][y] = true;
            for(int i = 1; i <= 9; i++){

                if(!col[y][i] && !row[x][i] && !sub[calNum(x, y)][i]){
                    board[x][y] = (char)(i + '0');
                    col[y][i] = true;
                    row[x][i] = true;
                    sub[calNum(x, y)][i] = true;
                    int[] newLoc = genLoc(x, y);
                    if(trackBack(board, newLoc[0], newLoc[1])){
                     return true;
                    }
                    
                    col[y][i] = false;
                    row[x][i] = false;
                    sub[calNum(x, y)][i] = false;
                } 

            }
            
            isVisited[x][y] = false;
        }else{
            int[] newLoc = genLoc(x, y);
            if(trackBack(board, newLoc[0], newLoc[1])){
                return true;
            }
        }
        return false;
       
        
    }
    
    private int calNum(int r, int c){
        
        r = r / 3;
        c = c / 3;
        
        return r * 3 + c;
        
    }
    
    private int[] genLoc(int x, int y){
        int[] res = new int[2];
        
        if(y + 1 < 9){
            res[0] = x;
            res[1] = y + 1;
        }else{
            res[0] = x + 1;
            res[1] = 0;
        }
        
        return res;
    }
    
}
```

### DFS

#### 1. 矩阵中的连通分量数

[200. Number of Islands](https://leetcode.com/problems/number-of-islands/)

```java
class Solution {
 
    private int[][] d = {{-1,0}, {0, 1}, {1,0}, {0, -1}};
    private int m;
    private int n;
    public int numIslands(char[][] grid) {
        int result = 0;
        m = grid.length;
        if(m == 0){
            return 0;
        }
        n = grid[0].length;
        boolean[][] isVisited = new boolean[m][n];
        
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                
                if(!isVisited[i][j] && grid[i][j] == '1'){
                    floodFill(grid, isVisited, i, j);
                    result++;
                }
            }
        }
        
        return result;
    }
    
    private void floodFill(char[][] grid, boolean[][] isVisited, int x, int y){
  
        isVisited[x][y] = true;
        for(int i = 0; i < 4; i++){
            int newX = x + d[i][0];
            int newY = y + d[i][1];
            
            if(inArea(newX, newY) && !isVisited[newX][newY] && grid[newX][newY] == '1'){
                floodFill(grid, isVisited, newX, newY);
            }
        }
 
    }
    
    private boolean inArea(int x, int y){
        return x >= 0 && x < m && y >= 0 && y < n;
    }
    
}
```

#### 2. 填充封闭区域

[130. Surrounded Regions](https://leetcode.com/problems/surrounded-regions/)

访问完外围的，剩下的就是封闭的

```java
class Solution {
    
    private int m;
    private int n;
    private int[][] d = {
        {-1, 0},
        {0, 1},
        {1, 0},
        {0, -1}
    };
    
    public void solve(char[][] board) {
        
        m = board.length;
        if(m == 0){
            return;
        }
        n = board[0].length;
        boolean[][] isVisited = new boolean[m][n];
        
        for(int i = 0; i < m; i++){
            
          
            if(board[i][0] == 'O' && !isVisited[i][0]){
                dfs(board, isVisited, i, 0);
            }else{
                isVisited[i][0] = true;
            }
            if(board[i][n-1] == 'O'  && !isVisited[i][n - 1]){
                dfs(board, isVisited, i, n - 1);
            }else{
                isVisited[i][n - 1] = true;
            }
        }
        
        
        for(int i = 0; i < n; i++){
            
           
            
            if(board[0][i] == 'O' && !isVisited[0][i]){
                dfs(board, isVisited, 0, i);
            }else{
                isVisited[0][i] = true;
            }
            if(board[m - 1][i] == 'O' && !isVisited[m - 1][i]){
                dfs(board, isVisited, m - 1, i);
            }else{
                isVisited[m - 1][i] = true;
            }
        }
        
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                if(!isVisited[i][j] && board[i][j] == 'O'){
                    board[i][j] = 'X';
                }
            }
        }
        
        
        
    }
    
    private void dfs(char[][] board, boolean[][] isVisited, int x, int y){
        
        
        isVisited[x][y] = true;

    
        for(int i = 0; i < 4; i++){
            int newX = x + d[i][0];
            int newY = y + d[i][1];
            if(inArea(newX, newY) && !isVisited[newX][newY] && board[newX][newY] == 'O'){
                dfs(board, isVisited, newX, newY);
            }

        }
        
    }
    
    private boolean inArea(int x, int y){
        return x >= 0 && x < m && y >= 0 && y < n;
    }
    
    

}
```

#### 3. 能到达太平洋和大西洋的区域

[417. Pacific Atlantic Water Flow](https://leetcode.com/problems/pacific-atlantic-water-flow/)

```java
class Solution {
    
    private int m;
    private int n;
    private int[][] d = {
        {-1, 0},
        {0, 1},
        {1, 0},
        {0, -1}
    };
    public List<List<Integer>> pacificAtlantic(int[][] matrix) {
        
        List<List<Integer>> res = new ArrayList<>();
    
        m = matrix.length;
        if(m == 0){
            return res;
        }
        n = matrix[0].length;
        
        boolean[][] canReachP = new boolean[m][n];
        boolean[][] canReachA = new boolean[m][n];
        
        for(int i = 0; i < m; i++){
            if(!canReachP[i][0])
                dfs(matrix, canReachP, i, 0);
            if(!canReachA[i][n - 1])
                dfs(matrix, canReachA, i, n - 1);
        }
        
        for(int i = 0; i < n; i++){
            if(!canReachP[0][i])
                dfs(matrix, canReachP, 0, i);
            if(!canReachA[m - 1][i])
                dfs(matrix, canReachA, m - 1, i);
        }
        
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                if(canReachA[i][j] && canReachP[i][j])
                    res.add(Arrays.asList(i, j));
            }
        }
        
        
        return res;
        
    }
    
    private void dfs(int[][] matrix, boolean[][] canReach, int x, int y){
        
        canReach[x][y] = true;
        
        for(int i = 0; i < 4; i++){
            int newX = x + d[i][0];
            int newY = y + d[i][1];
            
            if(inArea(newX, newY) && !canReach[newX][newY] && matrix[x][y] <= matrix[newX][newY])
                dfs(matrix, canReach, newX, newY);
        }
        
    }
    
    
    private boolean inArea(int x, int y){
        return x >= 0 && x < m && y >= 0 && y < n;
    }
}
```



## 3. 排序

### 桶排序

#### 1. 出现频率最高的k个元素

[347. Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/)

设置若干桶，每个桶存放出现频率相同的元素，桶的下标即元素出现的频率，逆序输出桶的前k项即所求

* 利用桶排序

```java
public List<Integer> topKFrequent(int[] nums, int k) {
    Map<Integer, Integer> frequentMap = new HashMap<>(); 
    //统计频率
    for(int num : nums){
        frequentMap.put(num, frequentMap.getOrDefault(num, 0) + 1); 
    }

    List<Integer>[] bucket = new ArrayList[nums.length + 1]; //桶
    for(Integer i : frequentMap.keySet()){

        int count = frequentMap.get(i);

        if(bucket[count] == null){
            bucket[count] = new ArrayList<>();
        }
        bucket[count].add(i);

    }

    List<Integer> result = new ArrayList<>();

  	//取出前k个元素
    for(int i = nums.length ; i >=0 && result.size() < k; i--){
        if(bucket[i] != null){
            result.addAll(bucket[i]);
        }
    }
    return result;

}
```

* 利用堆排序

```java
import javafx.util.Pair;

class Solution {
    public List<Integer> topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        
        for(int num : nums)
            map.put(num ,map.getOrDefault(num, 0)+1);
        
        PriorityQueue<Pair<Integer, Integer>> queue = new PriorityQueue<Pair<Integer, Integer>>((p1, p2) -> {
            return p1.getKey() - p2.getKey();
        });
        
        final int size = k;
        
        map.forEach((K,v) -> {
            
            queue.offer(new Pair(v,K));
            if(queue.size() == size + 1){
                queue.poll();
            }
        });
        
        List<Integer> list = new ArrayList<>();
        while(!queue.isEmpty()){
            list.add(queue.poll().getValue());
        }
        
            
        
        return list;     
    }
}
```



#### 2. 按照字符出现次数对字符串排序

[451. Sort Characters By Frequency](https://leetcode.com/problems/sort-characters-by-frequency/)

思路：先统计字符出现次数，再按出现次数进行桶排序，最后逆序遍历桶生成字符串

```java
public String frequencySort(String s) {
    int len = s.length();
    Map<Character, Integer> map = new HashMap<>();

    for(char c : s.toCharArray())
        map.put(c, map.getOrDefault(c, 0) + 1);

    //桶排序
    List<Character>[] bucket = new ArrayList[len + 1];

    for(char c : map.keySet()) {
        int freq = map.get(c);
        if(bucket[freq]  == null) {
            bucket[freq] = new ArrayList<>();
        }
        bucket[freq].add(c);

    }

    //逆序遍历桶
    StringBuilder strBuilder = new StringBuilder();
    for(int i = len; i > 0; i--) {

        int freq = i;
        if(bucket[i] != null) {
            bucket[i].forEach( c -> {
                for(int j = 0; j < freq; j++) {
                    strBuilder.append(c);
                }
            });
        }

    }

    return strBuilder.toString();
```



### 堆

#### 1. kth Element

[215. Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/)

* 快速选择（利用快排的划分）时间复杂度`O(N)`

  ```java
  public int findKthLargest(int[] nums, int k) {
      int low = 0, high = nums.length - 1;
      k = nums.length -k; //第k个大的数就是（长度-k）的数
  
      while(low < high ) {
          int j = partition(nums, low, high);
  
          if(j == k){
              break;
          }
          if(j > k) {
               high = j - 1;
          }else {
               low = j + 1;
          }
  
      }
      return nums[k];
  }
  
  
  private int partition(int[] nums, int low, int high){
      int i = low;
      int j = high + 1;
      int v = nums[low];
      while(true){
          while(nums[++i] < v) {
              if(i == high){
                  break;
              }
          }
  
          while(nums[--j] > v) {
              if(j == low){
                  break;
              }
          }
  
          if( i >= j) {
              break;                
          }
  
          swap(nums, i, j);
  
      }
      swap(nums, low, j);
  
      return j;
  }
  
  //交换函数
  private void swap(int[] nums, int i, int j){
      int temp = nums[i];
      nums[i] = nums[j];
      nums[j] = temp;
  }
  ```

* 堆，时间复杂度 O(NlogK)，空间复杂度 O(K)

  使用小顶堆，维持堆的大小为k，所有数据插入完，堆顶的元素即为第k个数

  ```java
  public int findKthLargest(int[] nums, int k) {
  
      PriorityQueue<Integer> queue = new PriorityQueue<>();
  
      for(int num : nums){
          queue.offer(num);
          if(queue.size() > k)
              queue.poll();
      }
  
      return queue.peek();
  }
  ```

  ### 荷兰国旗问题
  
  #### 1. 按颜色进行排序
  
  [75. Sort Colors](https://leetcode.com/problems/sort-colors/)
  
  思路：利用三路快排的思想
  
  ```java
  public void sortColors(int[] nums) {
      int zero = -1; //[0,zero] == 0
      int two = nums.length; //[two,nums.length - 1] == 2
      int i = 0; //[zero + 1,i-1]  == 1;
  
      while(i < two) {
          if(nums[i] == 1){
              i++;
          }else if(nums[i] == 2){
              swap(nums, i, --two);
  
          }else { 
              swap(nums, ++zero, i++);
  
          }
      }
  }
  
  private void swap(int[] nums, int i, int j) {
      if(i != j){
      int temp = nums[i];
      nums[i] = nums[j];
      nums[j] = temp;
      }
  }
  ```
  
  

## 4. 数学

### 1. 其它

#### 1.乘积数组

[238. Product of Array Except Self (Medium)](https://leetcode.com/problems/product-of-array-except-self/)

给定一个数组求其乘积数组，乘积数组每个元素等于原数组中除了当前元素之外所有元素的乘积

思路：当前元素的乘积等于该元素左边所有元素的乘积 * 右边所有元素的乘积

```java
public int[] productExceptSelf(int[] nums) {
    int len = nums.length;
    int[] result = new int[len];
    Arrays.fill(result,1);

  	//左边所有的乘积
    for(int i = 1; i < len; i++){
        result[i] = result[i - 1] * nums[i - 1];                     
    }

    int right = 1;
  	//再乘右边所有元素的乘积
    for(int i = len - 2 ; i >= 0; i--){
        right *= nums[i + 1];
        result[i] *= right;                     
    }

    return result;

}
```

### 2. 多数投票问题

#### 1. 数组中出现次数大于n/2的元素

[169. Majority Element](https://leetcode.com/problems/majority-element/)

用一个count来统计元素的出现次数，当与待统计元素不一样时count--，如果count==0，说明前i个元素中没有出现多余i/2。

```java
public int majorityElement(int[] nums) {

    int count = 0, result = 0;

    for(int num : nums){
        result = (count == 0) ? num : result;
        if (result == num) {
            count++;
        } else {
            count--;
        }
    }

    return result;

}
```

## 5. 动态规划

### 1.矩阵路径

#### 1.矩阵的总路径数

[62. Unique Paths(Medium)](https://leetcode.com/problems/unique-paths/)

```java
public int uniquePaths(int m, int n) {
    int[][] result = new int[m][n];

    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            if(i != 0 && j != 0){
                result[i][j] = result[i -1][j] + result[i][j - 1];
            }else{
                result[i][j] = 1;
            }
        }
    }

    return result[m-1][n-1];
}
```

优化空间复杂度

由`result[i][j] = result[i-1][j] + result[i][j-1]`看出只用了前一行的数据，所以`result[i] = result[i] + result[i-1]`实现如下：

```java
public int uniquePaths(int m, int n) {
    int[] result = new int[n];

    Arrays.fill(result, 1);
    for(int i = 1; i < m; i++){
        for(int j = 1; j < n; j++){
           result[j] += result[j-1];
        }
    }

    return result[n-1];
}
```

#### 2.矩阵最小路径和

[64. Minimum Path Sum](https://leetcode.com/problems/minimum-path-sum/)

```java
public int minPathSum(int[][] grid) {
    int row = grid.length;
    int column = grid[0].length;
    for(int i = 0; i < row; i++){
        for(int j = 0; j < column; j++){
            if(i == 0 && j == 0){
                continue;
            }

            if( i == 0) {
                grid[i][j] += grid[i][j-1];
            }else if( j == 0){
                grid[i][j] += grid[i-1][j];
            }else {
                grid[i][j] += Math.min(grid[i - 1][j], grid[i][j - 1]);                    
            }

        }
    }

    return grid[row - 1][column - 1];

}
```

#### 3.组成整数的最小平方数量

[279. Perfect Squares](https://leetcode.com/problems/perfect-squares/)

思路：dp[n]为组成n的最小平方数，dp[n] = min(dp[n], dp[n - i * i] 1 <= i*i < n

```java
public int numSquares(int n) {

    int[] memo = new int[n + 1];
    memo[1] = 1;

    for(int i = 2; i <= n; i++){
        int min = Integer.MAX_VALUE;
        for(int j = 1; j * j <= i; j++){
            min = Math.min(min, memo[i - j *j] + 1);
        }

        memo[i] = min;
    }
    return memo[n];
}
```



### 2. 斐波那契数列

#### 1.爬楼梯

[70. Climbing Stairs](https://leetcode.com/problems/climbing-stairs/)

思路：动态规划，状态定义dp[n]走到第n阶台阶总共有多少种办法，dp[n] = dp[n - 1] + dp[n -2]，初始状态dp[1]  =1 ,dp[0] = 1;

```java
public int climbStairs(int n) {
    if(n <= 2){
       return n;
    }

    int pre1 = 1, pre2 = 2, cur = 0;

    for(int i = 2; i < n; i++){
        cur = pre1 + pre2;
        pre1 = pre2;
        pre2 = cur;
    }

    return cur;     
}
```

#### 2.强盗抢劫

[198. House Robber](https://leetcode.com/problems/house-robber/)

思路：dp[n]强到第n个房子的最大收益，dp[n] = max(dp[n - 1], dp[n - 2] + nums[n]);

```java
public int rob(int[] nums) {
    if(nums.length == 0){
        return 0;
    }

    int[] memo = new int[nums.length];

    for(int j = 0; j < nums.length; j++){
        memo[j] = Math.max(j - 1 < 0 ? 0 : memo[j - 1] ,nums[j] + (j - 2 < 0 ? 0 : memo[j - 2]));  
    }




    return memo[nums.length - 1];
}
```

#### 3.强盗在环形街区抢劫

[213. House Robber II](https://leetcode.com/problems/house-robber-ii/)

思路：选择一个地方破除循环

```java
public int rob(int[] nums) {

    int n = nums.length;

    if(n == 0){
        return 0;
    }

    if(n == 1){
        return nums[0];
    }

    return Math.max(rob(nums, 0, n - 2), rob(nums, 1, n - 1));
}

private int rob(int[] nums, int low, int high) {

    int pre1 = 0, pre2 = 0;
    for(int i = low; i <= high; i++){
        int temp = pre1;
        pre1 = Math.max(pre1, pre2 + nums[i]);

        pre2 = temp;
    }

    return pre1;

}
```



### 3.分割整数

#### 1.分割整数的最大乘积

[343. Integer Break](https://leetcode.com/problems/integer-break/)

思路：状态定义：dp[n]分割n的最大乘积，dp[n] = Max(dp[n], i * n - i, i * dp[n - i]) 1 <= i < n

```java
public int integerBreak(int n) {
    int[] memo = new int[n+1];
    memo[1] = 1;

    for(int i = 2; i <= n; i++){

        for(int j = 1; j < i; j++){
            memo[i] = max3(memo[i], j * (i - j), j * memo[i - j]);
        }

    }

    return memo[n];
}


private int max3(int a, int b, int c){
    return Math.max(a, Math.max(b, c));
}
```

#### 2.分割整数构成字符串

[91. Decode Ways](https://leetcode.com/problems/decode-ways/)

思路：dp[n]前n个数字可以解码成的字符串的个数，dp[n] = dp[n - 1] + dp[ n - 2] 

```java
public int numDecodings(String s) {

    int n = s.length();
    int[] memo = new int[n + 1];
    memo[0] = 1;
    memo[1] = s.charAt(0) == '0' ? 0 : 1;

    for(int i = 2; i <= n; i++){

        int two = Integer.valueOf(s.substring(i - 2, i));

        if(s.charAt(i - 1) != '0'){
            memo[i] += memo[i - 1];

        }

        if(two >= 10 && two <= 26){
            memo[i] += memo[i - 2];
        }

    }

    return memo[n];

}
```

​                   

### 4.股票交易

#### 1.需要冷却的股票交易

[309. Best Time to Buy and Sell Stock with Cooldown](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

思路：sell[n],buy[n],res[n];分别表示以三种状态结尾的前n天的最大收益，

sell[n] = max(buy[n-1] + prices[n], sell[n - 1]); 

buy[n] = max(res[n -1] - prices[n], buy[n - 1]);

res[n] = max(res[n - 1], buy[n - 1], sell[ n - 1])

经过推理：res[n] = sell[n - 1];

```java
public int maxProfit(int[] prices) {
    int n = prices.length;
    if(n == 0) {
        return 0;
    }
    int[] sell = new int[n];
    int[] buy = new int[n];

    for(int i = 0; i < n; i++){
        buy[i] = Math.max(i - 2 < 0 ? -prices[i] : sell[i - 2] - prices[i], i - 1 < 0 ? Integer.MIN_VALUE : buy[i - 1]);
        sell[i] = Math.max(i - 1 < 0 ? 0 : buy[i - 1] + prices[i], i - 1 < 0 ? 0 : sell[i - 1]);
    }

    return Math.max(buy[n - 1], sell[n - 1]);
}
```

### 5.背包问题

#### 1.划分数组为相等的两部分

[416. Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/)

思路：把问题转化为在数组中求和为sum/2的部分

```java
public boolean canPartition(int[] nums) {


    int target = sum(nums), n = nums.length;
    if(target % 2 != 0){
        return false;
    }

    target /= 2;

    boolean[] result = new boolean[target + 1];

    for(int i = 1; i <= target; i++){
        result[i] = nums[0] == i;
    }

    for(int i = 1; i < n; i++){
        for(int j = target; j >= nums[i]; j--){
           result[j] = result[j - nums[i]] || result[j];
        }
    }

    return result[target];
}

private int sum(int[] nums){
    int result = 0;
    for(int temp : nums)
        result += temp;
    return result;
}
```

#### 2.找零钱最少的硬币数

[322. Coin Change](https://leetcode.com/problems/coin-change/)

思路：完全背包问题，F(i,c) = max(F(i-1, c), F(i, c-wi ) + vi);

```java
public int coinChange(int[] coins, int amount) {
    int[] result = new int[amount + 1];
    int n = coins.length;

    for(int i = 0; i <= amount; i++)
        result[i] = i % coins[0] == 0 ? i / coins[0] : Integer.MAX_VALUE;

    for(int i = 1; i < n; i++){
        for(int j = coins[i]; j <= amount; j++){
            if(result[j - coins[i]] != Integer.MAX_VALUE){
                result[j] = Math.min(result[j], result[j - coins[i]] + 1);
            }
        }
    }

    return result[amount] == Integer.MAX_VALUE ? -1 : result[amount];
}
```

#### 3.组合总和

[377. Combination Sum IV](https://leetcode.com/problems/combination-sum-iv/)

思路：涉及顺序的完全背包问题，物品的迭代放到里层，f(c) = sum(f(c - i))

```java
public int combinationSum4(int[] nums, int target) {

    int[] memo = new int[target + 1];
    memo[0] = 1;

    for(int i = 1; i <= target; i++){
        for(int j = 0; j < nums.length; j++){
            if(i - nums[j] >= 0){
                memo[i] += memo[i - nums[j]];
            }
        }
    }
    return memo[target];
}
```

#### 4.字符串按单词列表分割

[139. Word Break](https://leetcode.com/problems/word-break/)

思路：涉及顺序的完全背包问题, dp[i] = dp[ i ] || dp[i - len]

```java
public boolean wordBreak(String s, List<String> wordDict) {
    int n = s.length();

    boolean[] memo = new boolean[n+1];
    memo[0] = true;

    for(int i = 1; i <= n; i++){
        for(String word : wordDict){
            int len = word.length();
            if(len <= i && word.equals(s.substring(i - len, i))){
                memo[i] = memo[i] || memo[i - len];
            }
        }
    }

    return memo[n];

}
```

#### 5.改变一组数的正负号使得它们的和为一给定的正数

[494. Target Sum](https://leetcode.com/problems/target-sum/)

思路：转化为求找到和为（sum + target)/2的背包问题,f(i,c) = f(i - 1, c) + f(i, c - nums[i]);

```java
public int findTargetSumWays(int[] nums, int S) {

    int sum = sum(nums);

    int target = (sum + S) / 2;
    if(sum < S || (sum + S) % 2 != 0){
        return 0;
    }

    int[] memo = new int[target + 1];
    memo[0] = 1;

    for(int i = 0; i < nums.length; i++){
        for(int j = target; j >= nums[i]; j--){
            memo[j] = memo[j] + memo[j - nums[i]];
        }
    }


    return memo[target];
}

private int sum(int[] nums){
    int res = 0;
    for(int num : nums){
        res += num;
    }

    return res;
}
```

### 6.最长递增子序列LIS

#### 1.最长递增子序列

[300. Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)

思路：dp[n]为包括第n个数的最长递增子序列，dp[n] = max(dp[i] + 1) 0< i < n

```java
public int lengthOfLIS(int[] nums) {

    int n = nums.length, result = 0;
    int[] memo = new int[n];

    for(int i = 0; i < n; i++){

        memo[i] = 1;
        for(int j = 0; j < i; j++){
            if(nums[i] > nums[j]){
                memo[i] = Math.max(memo[i], memo[j] + 1);
            }
        }

        result = Math.max(result, memo[i]);
    }

    return result;
}
```

#### 2.最长摆动子序列

[376. Wiggle Subsequence](https://leetcode.com/problems/wiggle-subsequence/)

思路：down[n] = up[n-1] +1, up[n] = down[n - 1] + 1 

```java
public int wiggleMaxLength(int[] nums) {

    int n = nums.length;

    if(n == 0){
        return 0;
    }

    int down = 1, up = 1;

    for(int i = 1; i < n; i++){
        if(nums[i] > nums[i - 1]){
           up = down + 1;
        }else if(nums[i] < nums[i -1]){
           down = up + 1;
        }
    }

    return Math.max(up, down);

}
```

### 7.字符串

#### 1.编辑距离

[72. Edit Distance](https://leetcode.com/problems/edit-distance/)

思路：`dp[m][n]表示word1(0~m)==>word2(0~n)的编辑距离, dp[i][j - 1]表示插入, dp[i - 1][j]表示删除，dp[i - 1][j -1]表示替换`

```java
public int minDistance(String word1, String word2) {

    int m = word1.length(), n = word2.length();

    int[][] dp = new int[m + 1][n + 1];

    for(int i = 1; i <= n; i++)
        dp[0][i] = i;

    for(int i = 1; i <= m; i++)
        dp[i][0] = i;


    for(int i = 1; i <= m; i++){
        for(int j = 1; j <= n; j++){

            if(word1.charAt(i - 1) == word2.charAt(j - 1)){
                dp[i][j] = dp[i - 1][j - 1];
            }else {
                dp[i][j] = Math.min(dp[i - 1][j - 1], Math.min(dp[i][j - 1], dp[i - 1][j])) + 1;
            }
        }
    }

    return dp[m][n];

}
```

#### 2.最长公共子序列

[1143. Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/)

`dp[i][j]表示text1（0..i)text2(0..j)的最长公共子序列`

```java
public int longestCommonSubsequence(String text1, String text2) {
    int len1 = text1.length(), len2 = text2.length();
    int[][] dp = new int[len1 + 1][len2 + 1];

    for(int i = 1; i <= len1; i++){
        char c1 = text1.charAt(i - 1);
        for(int j = 1; j <= len2; j++){
            if(c1 == text2.charAt(j - 1)){
                dp[i][j] = dp[i - 1][j - 1] + 1;
            }else {
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }
    return dp[len1][len2];
}
```

#### 3.最长回文子序列

[516. Longest Palindromic Subsequence](https://leetcode.com/problems/longest-palindromic-subsequence/)

`两种解法：第一种，求序列与相反序列的最长公共子序列，第二种，区间动态规划：dp[i][j]表示从i-j最长回文子序列的长度`

```java
public int longestPalindromeSubseq(String s) {
    int len = s.length();
    int[][] dp = new int[len][len];

  	//一个字符回文子序列长度为1
    for(int i = 0; i < len; i++){
        dp[i][i] = 1;
    }

    for(int i = len - 2; i >= 0; i--){
        char c = s.charAt(i);
        for(int j = i + 1; j < len; j++){
            if(c == s.charAt(j)){
                dp[i][j] = dp[i + 1][j - 1] + 2;
            }else {
              	//单独考虑一个字符串加入其中
                dp[i][j] = Math.max(dp[i][j - 1], dp[i + 1][j]);
            }
        }
    }
    return dp[0][len - 1];
}
```



## 6. 双指针

### 1. 有序数组的two sum

[167. Two Sum II - Input array is sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/)

思路：使用双指针，当值大于target时尾指针前移，小于时头指针前移，等于时直接返回

```java
public int[] twoSum(int[] numbers, int target) {
    int i = 0, j = numbers.length - 1;

    while(i < j) {
        if(numbers[i] + numbers[j] == target){
           return new int[]{i+1, j+1};
        }else if(numbers[i] + numbers[j] > target) {
            j--;                
        } else {
            i++;
        }
    }
    return new int[2];
}
```

### 2. 反转字符串中的元音字符

[345. Reverse Vowels of a String](https://leetcode.com/problems/reverse-vowels-of-a-string/)

思路：使用双指针，当两个指针都遇到元音字母时停下，交换

```java
public String reverseVowels(String s) {
    Set<Character> vowels = new HashSet<>(Arrays.asList('a','e','i','o','u','A','E','I','O','U'));
    char[] array = s.toCharArray();
    int i = 0, j = array.length - 1;

    while(i < j){
        if(!vowels.contains(array[i])) {
           i++;
        }else if(!vowels.contains(array[j])) {
            j--;
        }else{
            swap(array, i++, j--);
        }

    } 

    return new String(array);

}

private void swap(char[] s, int i, int j){
    char temp = s[i];
    s[i] = s[j];
    s[j] = temp;
}
```

### 3. 最小长度子数组sum

[209. Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/)

思路：使用滑动窗口，当前窗口内元素之和小于sum，增大窗口，否则减小窗口

```java
public int minSubArrayLen(int s, int[] nums) {
    int l = 0, r = -1, sum = 0, result = nums.length + 1; //[l,r]为窗口
    while(l < nums.length) {

        if (sum < s && r + 1 < nums.length ) {
            sum += nums[++r];
        } else  {
            sum -= nums[l++];
        }

        if (sum >= s) {
            result = Math.min(result, r - l + 1);
        }

    }

    return result == nums.length + 1 ? 0 : result;
}
```

### 4. 没有重复字符的最长子串

[3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)

思路：使用滑动窗口，使窗口内不包含重复子串

```java
public int lengthOfLongestSubstring(String s) {
    boolean[] repeat = new boolean[256];

    int l = 0, r = -1, result = 0, len = s.length();


    while(l < len){
        if(r + 1 < len && !repeat[s.charAt(r + 1)]){
            repeat[s.charAt(++r)] = true;
        } else {
            repeat[s.charAt(l++)] = false;

        }

        result = Math.max(result, r - l + 1);
    }


    return result;
}
```

## 7. BFS

广度优先搜索，可以解决无权图的最短路径问题，在程序实现广度优先搜索时注意如下两个问题：

* 用队列存储每一轮遍历得到的结点
* 对于遍历过的节点，应将它标记防止重复访问

### 1. 组成整数的最小平方数

[279. Perfect Squares](https://leetcode.com/problems/perfect-squares/)

```java
public int numSquares(int n) {

    Queue<Integer> queue = new LinkedList<>();
    boolean[] flag = new boolean[n + 1];

    queue.add(n);
    flag[n] = true;
    int step = 0;

    while(!queue.isEmpty()) {
        int size = queue.size();
        step++;
        while(size-- > 0) {
            int num =  queue.poll();
            for(int i = 1; ; i++) {
                int temp = num - i * i;
                if(temp < 0) {
                    break;
                }

                if(temp == 0) {
                    return step;
                }

                if(!flag[temp]) {
                    flag[temp] = true;
                    queue.add(temp);
                }

            }
        }
    }

    return 0;     

}
```

### 2. 最短单词路径

[127. Word Ladder](https://leetcode.com/problems/word-ladder/)

```java
public int ladderLength(String beginWord, String endWord, List<String> wordList) {
    wordList.add(beginWord);

    int n = wordList.size();
    int start = n - 1, end = 0;

    while(end < n && !wordList.get(end).equals(endWord))
        end++;

    if(end == n){
        return 0;
    }


    return getShrotestPath(buildGraphic(wordList), start, end);
}


private List<List<Integer>> buildGraphic(List<String> wordList) {

    List<List<Integer>> graphic = new ArrayList<>();
    int n = wordList.size();

    for(int i = 0; i < n; i++) {
        List<Integer> list = new ArrayList<>();
        for(int j = 0; j < n; j++) {
            if(isConnect(wordList.get(i), wordList.get(j)))
                list.add(j);

        } 
        graphic.add(list);
    }

    return graphic;
}

private boolean isConnect(String s1, String s2) {
    int diff = 0;
    for(int i = 0; i < s1.length() && diff <= 1; i++){
        if(s1.charAt(i) != s2.charAt(i)) 
            diff++;
    }
    return diff == 1;
}

private int getShrotestPath(List<List<Integer>> graphic, int start, int end) {

    Queue<Integer> queue = new LinkedList<>();
    boolean[] flag = new boolean[graphic.size()];
    int step = 1;
    queue.offer(start);
    flag[start] = true;

    while(!queue.isEmpty()) {
        step++;

        int size = queue.size();
        while(size-- > 0) {
            int cur = queue.poll();

            for(int temp : graphic.get(cur)) {
                if(temp == end) {
                    return step;
                }

                if(!flag[temp]) {
                    queue.add(temp);
                    flag[temp] = true;

                }
            }

        }

    }
    return 0;
}
```



# 数据结构相关

## 1. 树

### 1. 归并两棵树

[617.MergeTwoBinaryTrees(Easy)](https://leetcode.com/problems/merge-two-binary-trees/)

``` java
public TreeNode mergeTrees(TreeNode t1, TreeNode t2) {

   if(t1 == null && t2 == null){
       return null;
   }

    if(t1 == null){
        return t2;
    }else if(t2 == null){
        return t1;
    }

    //t1做为新树的根节点
    t1.val += t2.val;
    t1.left = mergeTrees(t1.left, t2.left);
    t1.right = mergeTrees(t1.right, t2.right);

    return t1;

}
```

### 2. 翻转树

[226. Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/)

```java
public TreeNode invertTree(TreeNode root) {
    if(root == null){
        return null;
    }

    TreeNode temp = root.left; //保存左节点
    root.left = invertTree(root.right);
    root.right = invertTree(temp);
    return root;
}
```

### 3. 中序遍历

[94. Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)

- 递归实现

  ```java
  class Solution {
    private List<Integer> result = new ArrayList<>();//利用成员变量来放遍历顺序
    public List<Integer> inorderTraversal(TreeNode root) {
        if(root != null){    
            inorderTraversal(root.left);
            result.add(root.val);
            inorderTraversal(root.right);
        }
        return result;        
    }
  }
  ```

- 非递归实现

  ```java
  public List<Integer> inorderTraversal(TreeNode root) {
      List<Integer> result = new ArrayList<>();
  
      TreeNode current = root;
      LinkedList<TreeNode> stack = new LinkedList<>();
      while(current != null || !stack.isEmpty()){
          while(current != null){
              stack.addFirst(current);
              current = current.left;              
          }
          current = stack.removeFirst();
          result.add(current.val);
          current = current.right;
      }
      return result;
  
    }
  ```
  
- 仿系统递归栈的实现

  ```java
  class Command {
      String  cmd; //go,print
      TreeNode node;
      
      Command(String cmd, TreeNode node) {
          this.cmd = cmd;
          this.node = node;
          
      }
      
  }
  
  
  class Solution {
      public List<Integer> inorderTraversal(TreeNode root) {
          List<Integer> res = new ArrayList<>();
          LinkedList<Command> stack = new LinkedList<>();
          
          if(root == null) {
              return res;
          }
          
          stack.push(new Command("go", root));
          
          while(!stack.isEmpty()) {
              Command command = stack.pop();
              
              if(command.cmd.equals("print")) {
                  res.add(command.node.val);                
              } else {
                  if(command.node.right != null) {
                      stack.push(new Command("go", command.node.right));
                  }
                  stack.push(new Command("print", command.node));
                  if(command.node.left != null) {
                      stack.push(new Command("go", command.node.left));
                  }
                                 
              }
              
              
          }
          
          return res;   
      }
  }
  ```

  

### 4. 间隔遍历

[337. House Robber III](https://leetcode.com/problems/house-robber-iii/)

不能抢直接相连的两个结点，所以在rob(root)和rob(root.left) + rob(root.right) 里面取较大的

```java
public int rob(TreeNode root) {
    if(root == null){
        return 0;
    }

    int val1 = root.val;

    if(root.left != null){
        val1 += rob(root.left.left) + rob(root.left.right);
    }

     if(root.right != null){
        val1 += rob(root.right.left) + rob(root.right.right);
    }

    return Math.max(val1 , rob(root.left) + rob(root.right));

}
```

### 5. 两节点之间最长路径

[543. Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/)

思路：结点的最长路径等于左右子树的高度之和

```java
private int max = 0;

public int diameterOfBinaryTree(TreeNode root) {

    deepth(root);
    return max;

}

private int deepth(TreeNode root){
    if(root == null) {
        return 0;
    }

    int rightDeepth = deepth(root.right);
    int leftDeepth = deepth(root.left);

    max = Math.max(max, rightDeepth + leftDeepth);


    return Math.max(rightDeepth,leftDeepth) + 1;
}
```

### 6. 把二叉查找树每个节点的值都加上比它大的节点的值

[538. Convert BST to Greater Tree](https://leetcode.com/problems/convert-bst-to-greater-tree/)

思路：中序遍历的反方向

```java
private int sum = 0;

public TreeNode convertBST(TreeNode root) {

    convert(root);

    return root;        
}


private void convert(TreeNode current) {

    if(current == null) {
        return;
    }

    convert(current.right);
    current.val += sum;
    sum = current.val;
    convert(current.left);

}
```

### 7. 二叉树的先序遍历

[144. Binary Tree Preorder Traversal](https://leetcode.com/problems/binary-tree-preorder-traversal/)

思路：利用栈模拟系统的递归栈

```java
class Command {
    String  cmd; //go,print
    TreeNode node;
    
    Command(String cmd, TreeNode node) {
        this.cmd = cmd;
        this.node = node;
        
    }
    
}

class Solution {
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        LinkedList<Command> stack = new LinkedList<>();
        
        if(root == null) {
            return res;
        }
        
        stack.push(new Command("go", root));
        
        while(!stack.isEmpty()) {
            Command command = stack.pop();
            
            if(command.cmd.equals("print")) {
                res.add(command.node.val);                
            } else {
                if(command.node.right != null) {
                    stack.push(new Command("go", command.node.right));
                }
                if(command.node.left != null) {
                    stack.push(new Command("go", command.node.left));
                }
               
                stack.push(new Command("print", command.node));
                
                
                
            }
            
            
        }
        
        return res;     
        
    }
}
```

### 8. 二叉树的后续遍历

[145. Binary Tree Postorder Traversal](https://leetcode.com/problems/binary-tree-postorder-traversal/)

```java
class Command {
    String  cmd; //go,print
    TreeNode node;
    
    Command(String cmd, TreeNode node) {
        this.cmd = cmd;
        this.node = node;
        
    }
    
}

class Solution {
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        LinkedList<Command> stack = new LinkedList<>();
        
        if(root == null) {
            return res;
        }
        
        stack.push(new Command("go", root));
        
        while(!stack.isEmpty()) {
            Command command = stack.pop();
            
            if(command.cmd.equals("print")) {
                res.add(command.node.val);                
            } else {
                stack.push(new Command("print", command.node));
                if(command.node.right != null) {
                    stack.push(new Command("go", command.node.right));
                }
                if(command.node.left != null) {
                    stack.push(new Command("go", command.node.left));
                }
               
                
            }            
            
        }
        
        return res;     
    }
}
```

### 9. 二叉树的层次遍历

[102. Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)

```java
public List<List<Integer>> levelOrder(TreeNode root) {

    List<List<Integer>> res = new ArrayList<>();
    Queue<TreeNode> queue =  new LinkedList<>();

    if(root ==  null){
        return res;
    }

    queue.offer(root);

    while(!queue.isEmpty()){

        int num = queue.size();

        List<Integer> list = new ArrayList<>();
        for(int  i = 0; i < num; i++) {
            TreeNode node = queue.poll();
            list.add(node.val);
            if(node.left != null) {
                queue.offer(node.left);
            }

            if(node.right != null) {
                queue.offer(node.right);
            }
        }
        res.add(list);


    }

    return res;  
}
```

### 10. 树的高度

[104. Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)

```java
public int maxDepth(TreeNode root) {
    if(root == null){
        return 0;
    }

    return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
}
```

### 11. 从根结点到叶子结点的最短路径

[111. Minimum Depth of Binary Tree](https://leetcode.com/problems/minimum-depth-of-binary-tree/)

```java
public int minDepth(TreeNode root) {

    if(root == null) {
        return 0;
    }

  	//找到叶子结点
    if(root.left == null && root.right == null) {
        return 1;
    }

    if(root.left == null) {
        return minDepth(root.right) + 1;
    }


    if(root.right == null) {
        return minDepth(root.left) + 1;
    }

    int minLeft = minDepth(root.left);
    int minRight = minDepth(root.right);

    return Math.min(minLeft,minRight) + 1;

}
```

### 12. 树的对称

[101. Symmetric Tree](https://leetcode.com/problems/symmetric-tree/)

```java
public boolean isSymmetric(TreeNode root) {

    if(root == null)  {
        return true;
    }

    return isSymmetric(root.left, root.right);
}


private boolean isSymmetric(TreeNode r1, TreeNode  r2) {
    if(r1 == null && r2 == null) {
        return true;
    }

    if(r1 == null || r2 == null) {
        return false;
    }

    if(r1.val != r2.val) {
        return false;
    }

    return isSymmetric(r1.right,r2.left) && isSymmetric(r1.left,r2.right);

}
```

### 13. 平衡树

[110. Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/)

```java
public boolean isBalanced(TreeNode root) {

    if(root == null) {
       return true;
    }

    int result = deepth(root.left) - deepth(root.right);
    if(Math.abs(result) > 1){
        return false;
    }


    return isBalanced(root.left) && isBalanced(root.right);
}

private int deepth(TreeNode root) {
    if(root == null) {
        return 0;
    }

    return Math.max(deepth(root.left), deepth(root.right)) + 1;
}
```

### 14. 判断路径和是否等于一个数

[112. Path Sum](https://leetcode.com/problems/path-sum/)

```java
public boolean hasPathSum(TreeNode root, int sum) {

    if(root == null) {
        return false;
    }

    if( root.left == null && root.right == null){
        return root.val == sum;
    }

    return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val);

}
```

### 15. 统计左叶子结点的和

[404. Sum of Left Leaves](https://leetcode.com/problems/sum-of-left-leaves/)

```java
public int sumOfLeftLeaves(TreeNode root) {

    if(root == null) {
        return 0;
    }

    if(root.left == null) {
        return sumOfLeftLeaves(root.right);
    }

    if(root.left != null && root.left.left == null && root.left.right == null) {
        return root.left.val + sumOfLeftLeaves(root.right);
    }

    return sumOfLeftLeaves(root.left) + sumOfLeftLeaves(root.right);

}
```

### 16. 输出二叉树中所有从根到叶子的路径

[257. Binary Tree Paths](https://leetcode.com/problems/binary-tree-paths/)

```java
public List<String> binaryTreePaths(TreeNode root) {

    List<String> result = new ArrayList<>();
    if(root == null) {
        return result;
    }

    if(root.left == null && root.right == null) {
        result.add(Integer.toString(root.val));
        return result;
    }


    List<String> temp = binaryTreePaths(root.left);
    for(String s : temp){
        result.add(root.val + "->" + s);
    }

    temp = binaryTreePaths(root.right);
    for(String s : temp){
        result.add(root.val + "->" + s);
    }

    return result;

}
```

### 17. 统计路径和等于一个数的路径数量

[437. Path Sum III](https://leetcode.com/problems/path-sum-iii/)

```java
public int pathSum(TreeNode root, int sum) {
    if(root == null){
        return 0;
    }

    return findPath(root, sum) + pathSum(root.left, sum) + pathSum(root.right, sum);
}

//在以root为更结点的树中查找路径和为sum的路径，包含root
private int findPath(TreeNode root, int sum) {
    if(root == null) {
        return 0;
    }

    int result = 0;

    if(root.val == sum) {
        result++;
    }


    return findPath(root.left, sum - root.val) + findPath(root.right, sum - root.val) + result;
}
```

### 18. 二叉查找树的最近公共祖先结点

[235. Lowest Common Ancestor of a Binary Search Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)

```java
public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {

    if(root == null){
        return null;
    }

    if(p.val < root.val && q.val < root.val)
        return lowestCommonAncestor(root.left, p, q);

    if(p.val > root.val && q.val > root.val)
        return lowestCommonAncestor(root.right, p, q);

    return root;
}
```

### 19. 从有序数组构造二叉查找树

[108. Convert Sorted Array to Binary Search Tree](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/)

二分查找的顺序构建二叉查找树

```java
public TreeNode sortedArrayToBST(int[] nums) {

    if(nums == null || nums.length <= 0){
        return null;
    }

    int start = 0, end = nums.length - 1;

    int mid = (start + end) / 2;

    TreeNode root = new TreeNode(nums[mid]);        
    root.left = sortedArrayToBST(Arrays.copyOfRange(nums,0,mid));
    root.right = sortedArrayToBST(Arrays.copyOfRange(nums,mid + 1,end+1));
    return root;

}
```

### 20. 寻找二叉查找树的第k个元素

[230. Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)

* 递归实现

  ```java
  public int kthSmallest(TreeNode root, int k) {
  
      int countL = size(root.left);
      if(countL == k - 1){
          return root.val;
      }else if(countL > k - 1){
          return kthSmallest(root.left, k);
      }else {
          return kthSmallest(root.right, k - 1 - countL);
      }
  
  }
  
  public int size(TreeNode x){
      if(x == null){
          return 0;
      }
  
      return size(x.left) + size(x.right) + 1;
  }
  ```

* 中序遍历实现

  ```java
  class Solution {
      
      private int cnt = 0;
      private int val;
      public int kthSmallest(TreeNode root, int k) {
          
          inOrder(root, k);
          return val;         
      }
      
      private void inOrder(TreeNode root, int k){
          if(root == null){
              return;
          }
          
          inOrder(root.left, k);
          cnt++;
          if(cnt == k){
              val = root.val;
              return;
          }
          inOrder(root.right, k);
      }
  }
  ```

### 21. 二叉树的最近公共祖先

[236. Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)

```java
public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    if(root == null || root.val == p.val || root.val == q.val)
        return root;


    TreeNode left = lowestCommonAncestor(root.left, p, q);
    TreeNode right = lowestCommonAncestor(root.right, p, q);

    return left == null ? right : right == null ? left : root;        
}
```

## 2. 位运算

### 1. 统计0～n每个数的二进制表示中1的多少

[338. Counting Bits(Medium)](https://leetcode.com/problems/counting-bits/)

`i & i-1` 可以去除i的二进制表示中最后的1，如6(110) & 5 (101) = 4(100),所以`result[6] = result[4] + 1`

```java
public int[] countBits(int num) {
     int[] result = new int[num+1];
     int len = num + 1;
     for(int i = 1; i < len; i++){
         result[i] = result[i & (i - 1)] + 1;
     }
     return result;
 }
```

### 2. 找到数组中唯一不重复的数

[136. Single Number(Easy)](https://leetcode.com/problems/single-number/)

利用`x ^ x == 0`对所有数进行异或操作，最后剩下的，就是没有重复的那个数

```java
public int singleNumber(int[] nums) {
      int result = 0;
      for(int i : nums){
          result ^= i;
      }
      return result;

  }
```

## 3. 栈和队列

### 1. 数组中下一个比当前元素大的元素的距离

[739. Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)

遍历数组时把数组用栈保存起来，如果当前元素大于栈顶元素则说明距栈顶元素最近的一个比它大的元素就是当前元素

```java
public int[] dailyTemperatures(int[] T) {
    LinkedList<Integer> stack = new LinkedList<>();
    int len = T.length;
    int[] result = new int[len];
    for (int courrent = 0; courrent < len; courrent++ ) {
        while(!stack.isEmpty() && T[courrent] > T[stack.getFirst()]) {
            int pre = stack.removeFirst();
            result[pre] = courrent - pre;

        }
        stack.addFirst(courrent);
    }

    return result;

}
```

## 4. 字符串

### 1. 回文子字符串的个数

[647. Palindromic Substrings](https://leetcode.com/problems/palindromic-substrings/)

对字符串逐位进行拓展统计回文串的个数，对奇数串和偶数串分别进行统计

```java
public int countSubstrings(String s) {
    int count = 0;
    int len = s.length();
    for(int i = 0; i < len; i++) {
        count += extendSubstrings(s,i,i);
        count += extendSubstrings(s,i,i + 1);
    }

    return count;

}

private int extendSubstrings(String s, int start, int end) {
    int count = 0;
    int len = s.length();
    while(start >= 0 && end < len && (s.charAt(start) == s.charAt(end))) {
        count++;
        start--;
        end++;
    }
    return count;
}
```

### 2. 两个字符串包含的字符是否完全相同

[242. Valid Anagram](https://leetcode.com/problems/valid-anagram/)

思路：用HashMap存储字符的出现次数，然后进行对比

```java
public boolean isAnagram(String s, String t) {
    int len = s.length();

    if(len != t.length()){
        return false;
    }
    Map<Character, Integer> map = new HashMap<>();


    for(int i = 0; i < len; i++) {
        if(map.get(s.charAt(i)) == null) {
            map.put(s.charAt(i), 1);
        } else {
            map.put(s.charAt(i), map.get(s.charAt(i)) + 1);
        }

    }

    for(int i = 0; i < len; i++) {
        if(map.get(t.charAt(i)) != null) {
            map.put(t.charAt(i), map.get(t.charAt(i)) - 1);
            if(map.get(t.charAt(i)) <= 0){
                map.remove(t.charAt(i));
            }
        } else {
            return false;              
        }

    }

    return map.isEmpty();
}
```

### 3. 字符串同构

[205. Isomorphic Strings](https://leetcode.com/problems/isomorphic-strings/)

思路：定义映射规则，把两个字符串映射成同一种形式，然后再判断字符串是否相等add -> 1,2,2; egg ->122

```java
public boolean isIsomorphic(String s, String t) {

    int len = s.length();

    int[] pres = new int[256];
    int[] pret = new int[256];
    for(int i  = 0; i < len; i++) {
        char sc = s.charAt(i), tc =  t.charAt(i);
        if(pres[sc] != pret[tc]){
            return  false;
        }

        pres[sc] = i + 1;
        pret[tc] = i + 1;            
    }
    return true;
}
```



## 5. 链表

### 1.反转链表

[206. Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)

* 头插法

  ```java
  public ListNode reverseList(ListNode head) {
  
      ListNode headNode = new ListNode(-1);//头结点
      while(head != null){
          ListNode next = head.next;
          head.next = headNode.next;
          headNode.next = head;
          head = next;   
      } 
  
      return headNode.next;
  }
  ```

* 递归

  ```java
  public ListNode reverseList(ListNode head) {
  
      if(head == null || head.next == null){
          return head;
      }
  
      ListNode next = head.next;
  
      ListNode newHead = reverseList(next);
  
      next.next = head;
      head.next = null;
  
      return newHead;
  }
  ```

### 2. 归并两个有序的列表

[21. Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)

```java
public ListNode mergeTwoLists(ListNode l1, ListNode l2) {

    if(l1 == null) {
        return l2;
    }

    if(l2 == null) {
        return l1;
    }

    if(l1.val < l2.val) {
        l1.next = mergeTwoLists(l1.next, l2);
        return l1;    
    }else {
        l2.next = mergeTwoLists(l1, l2.next);
        return l2;
    }

}
```

### 3. 从有序列表中删除重复结点

[83. Remove Duplicates from Sorted List](https://leetcode.com/problems/remove-duplicates-from-sorted-list/)

```java
public ListNode deleteDuplicates(ListNode head) {
    ListNode cur = head;

    while(cur != null){
       ListNode next = cur.next;
        if(next == null) {
            break;
        }
        if(cur.val ==  next.val) {
            cur.next = next.next;
        } else {
            cur = cur.next;            
        }
    }

    return head;
}
```

### 4. 链表元素按奇偶索引聚集

[328. Odd Even Linked List](https://leetcode.com/problems/odd-even-linked-list/)

思路：使用双索引，选定奇数头结点和偶数头结点，挨个遍历奇数结点和偶数结点建立链表

```java
public ListNode oddEvenList(ListNode head) {
    if(head == null){
        return head;
    }
    ListNode oddHead = head, evenHead = head.next, curOdd = head, curEven = evenHead;

    while(curOdd.next != null && curEven.next != null) {
        curOdd.next = curOdd.next.next;
        curOdd = curOdd.next;

        curEven.next = curEven.next.next;
        curEven = curEven.next;

    }

    curOdd.next = evenHead;

    return head;
}
```

### 5. 链表求和

[445. Add Two Numbers II](https://leetcode.com/problems/add-two-numbers-ii/)

思路：先将链表元素全部入栈，然后挨个计算值新建链表

```java
public ListNode addTwoNumbers(ListNode l1, ListNode l2) {

    LinkedList<Integer> stack1 = bulidStack(l1);
    LinkedList<Integer> stack2 = bulidStack(l2);

    int carry = 0;
    ListNode head = new ListNode(-1);

    while(!stack1.isEmpty() || !stack2.isEmpty() || carry != 0) {
        int x = stack1.isEmpty() ? 0 : stack1.pop();
        int y = stack2.isEmpty() ? 0 : stack2.pop();
        int sum = x + y + carry;

        ListNode cur = new ListNode(sum % 10);
        cur.next = head.next;
        head.next = cur;

        carry = sum / 10;

    }

    return head.next;        
}

private LinkedList<Integer> bulidStack(ListNode head) {
    ListNode cur = head;
    LinkedList<Integer> stack = new LinkedList<>();

    while(cur !=  null){
        stack.push(cur.val);
        cur = cur.next;
    }
    return stack;
}
```

### 6. 删除链表中的指定元素

[203. Remove Linked List Elements](https://leetcode.com/problems/remove-linked-list-elements/)

```java
public ListNode removeElements(ListNode head, int val) {
    ListNode dummyHead = new ListNode(-1);
    dummyHead.next =  head;

    ListNode cur = dummyHead;

    while(cur != null && cur.next != null) {
        if(cur.next.val ==  val) {
            cur.next = cur.next.next;
        } else {
            cur = cur.next;

        }

    }

    return dummyHead.next;
}
```

### 7. 交换链表中的相邻结点

[24. Swap Nodes in Pairs](https://leetcode.com/problems/swap-nodes-in-pairs/)

```java
public ListNode swapPairs(ListNode head) {

    ListNode dummyHead = new ListNode(-1);
    dummyHead.next = head;
    ListNode pre = dummyHead;

    while(pre != null && pre.next != null && pre.next.next != null){
        ListNode node1 = pre.next;
        ListNode node2 = node1.next;

        pre.next = node2;
        node1.next = node2.next;
        node2.next = node1;

        pre = node1;
    }

    return dummyHead.next;
}
```

### 8. 删除给定的结点

[237. Delete Node in a Linked List](https://leetcode.com/problems/delete-node-in-a-linked-list/)

思路：将后一个结点的值复制到当前结点，将后一个结点删除

```java
public void deleteNode(ListNode node) {
    node.val = node.next.val;
    node.next =  node.next.next;
}
```

### 9. 删除链表的倒数第n个结点

[19. Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)

思路：使用双指针，快的指针先走n+1步，当快指针走到结束时，慢指针指向第n-1个元素

```java
public ListNode removeNthFromEnd(ListNode head, int n) {

    ListNode dummyHead = new ListNode(-1);
    dummyHead.next = head;
    ListNode fast = dummyHead;
    ListNode slow = dummyHead;

    while(n >= 0) {
        fast = fast.next;
        n--;
    }

    while(fast != null){
        fast = fast.next;
        slow = slow.next;
    }

    slow.next = slow.next.next;

    return dummyHead.next;

}
```

### 10. 回文链表

[234. Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/)

思路：找到链表的中间位置，然后反转后半部分看是否相等

```java
public boolean isPalindrome(ListNode head) {

    if(head == null) {
        return true;
    }

    ListNode fast = head, slow = head, slowPre = slow;


    while(slow != null && fast != null && fast.next != null) {
        slowPre = slow;
        slow = slow.next;
        fast = fast.next.next;
    }

    slowPre.next = null;
    ListNode backHead = reverse(slow);

    ListNode curFront = head;
    ListNode curBack = backHead;

    while(curFront != null) {
        if(curFront.val != curBack.val) {
            return false;
        }

        curFront = curFront.next;
        curBack = curBack.next;

    }

    return true;
}

private ListNode reverse(ListNode head) {
    ListNode dummyHead = new ListNode(-1);

    while(head != null){
        ListNode cur = head;
        head = head.next;
        cur.next = dummyHead.next;
        dummyHead.next = cur;     
    }

    return dummyHead.next;
}
```



## 6. 数组与矩阵

### 1. 把数组中的0移到末尾

[283. Move Zeroes](https://leetcode.com/problems/move-zeroes/)

思路：把非0的数挨个往前移，把剩余的数用0填充

```java
public void moveZeroes(int[] nums) {

    int insertFlag = 0; //插入标记

    for(int num : nums){
        if(num != 0){
            nums[insertFlag++] = num;
        }
    }

    while(insertFlag < nums.length){
        nums[insertFlag++] = 0;
    }

}
```

### 2. 找出1～n数组中缺少的元素

[448. Find All Numbers Disappeared in an Array](https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/)

思路：`[1～n]-1`=`[0~n-1]`为数组的下标，`nums[nums[i]-1] = -nums[nums[i]-1] `遍历数组大于0的元素的下标即为缺少的数

```java
public List<Integer> findDisappearedNumbers(int[] nums) {

    List<Integer> ret = new ArrayList<>();
  
    for(int i = 0; i < nums.length; i++){
        int val = Math.abs(nums[i]) - 1;

        if(nums[val] > 0) {
            nums[val] = -nums[val];
        }
    }

  	//查找结果
    for(int i = 0; i < nums.length; i++){
        if(nums[i] > 0){
            ret.add(i + 1);
        }

    }

    return ret;

}
```

### 3. 找出数组中重复的元素，数值在1～n之间

[287. Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/)

思路：数组长度为n+1，1～n可以看成数组的下标，利用双指针，快的一次走两步，慢的一次走一步，当两个指针相遇说明存在环，重新从起点开始两个指针相遇说明，相遇的位置就是重复元素

```java
public int findDuplicate(int[] nums) {
    int slow = nums[0], fast = nums[slow];

    while(slow != fast){
        slow = nums[slow];//走一步
        fast = nums[nums[fast]];//走两步
    }

    slow = 0;
    while(slow != fast){
        slow = nums[slow];
        fast = nums[fast];
    }

    return slow;

}
```

## 7. 哈希表

### 1. 数组中两个数的和为给定值

[1. Two Sum](https://leetcode.com/problems/two-sum/)

思路：用HashMap存储元素的值和索引，在访问nums[i]时，查找targer - nums[i],找到了则返回，没找到则继续查找

```java
public int[] twoSum(int[] nums, int target) {
    Map<Integer, Integer> map = new HashMap<>();
    for(int i = 0; i < nums.length; i++) {
        int newTarget = target - nums[i];//待查找的元素
        if(map.containsKey(newTarget)){
            return new int[]{map.get(newTarget), i};
        } else {
            map.put(nums[i], i);
        }
    }

    return new int[2];

}
```

### 2. 4个数组中各取一个元素使得和为0

[454. 4Sum II](https://leetcode.com/problems/4sum-ii/)

思路：用HashMap存储两个数组所有可能的和，依次遍历前两个数组所有可能的和，查找-前两个数组所有可能的和

```java
public int fourSumCount(int[] A, int[] B, int[] C, int[] D) {
    Map<Integer, Integer> map = new HashMap<>();
    int result = 0;

    for(int i : C)
        for(int j : D)
            map.put(i+j, map.getOrDefault(i+j, 0) + 1); //所有可能的和

    for(int i : A)
        for(int j : B){
            int newTarget = 0 - i -j; //待查找和
            if(map.containsKey(newTarget)){
                result += map.get(newTarget);
            }

        }

    return result;

}
```

### 3. 到一个点距离相等的两个点的个数

[447. Number of Boomerangs](https://leetcode.com/problems/number-of-boomerangs/)

思路：以一个点为枢纽，计算其它点到该点的距离，将距离和点的个数存储在HashMap中，求组合即可

```java
public int numberOfBoomerangs(int[][] points) {

    int result = 0;
    for(int[] pointI : points){
        Map<Integer, Integer> map = new HashMap<>(points.length);
        for(int[] point : points) {
            if(pointI != point){
                int dis = dis(pointI, point);
                map.put(dis, map.getOrDefault(dis, 0) + 1);
            }                
        }

      	//统计个数
        for(int dis : map.keySet())
            result += (map.get(dis) * (map.get(dis) - 1));

    }
  
    return result;
}

//计算距离，不开方避免使用浮点数
private int dis(int[] point1, int[] point2){
    return (point1[0] - point2[0]) * (point1[0] - point2[0])
        + (point1[1] - point2[1]) * (point1[1] - point2[1]);        
}
```

### 4. 给定下标范围是否有重复元素

[220. Contains Duplicate II](https://leetcode.com/problems/contains-duplicate-ii/)

思路：利用HashSet，固定set的大小，保持set的大小为范围-1

```java
public boolean containsNearbyDuplicate(int[] nums, int k) {

    HashSet<Integer> set = new HashSet<>(); //set的size即为滑动窗口

    for(int i = 0; i < nums.length; i++){
        if(!set.add(nums[i]))
            return true;
        if(set.size() == k+1)
            set.remove(nums[i - k]);


    }
    return false;
}
```

### 5. 判断数组是否有重复元素

[217. Contains Duplicate](https://leetcode.com/problems/contains-duplicate/)

```java
public boolean containsDuplicate(int[] nums) {
    Set<Integer> set = new HashSet<>();

    for(int num : nums)
        if(!set.add(num))
            return true;
    return false;
}
```

### 6. 给定下标范围，给定元素范围是否有重复元素

[220. Contains Duplicate III](https://leetcode.com/problems/contains-duplicate-iii/)

思路：利用SortedSet存放元素，便于查找指定范围内的元素

```java
public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
    SortedSet<Long> set = new TreeSet<>();

  	//t < 0 范围为负，不存在这样的元素
    if(t < 0){
        return false;
    }

    for(int i = 0; i< nums.length; i++) {
        if(!set.subSet((long)nums[i] - t, (long)nums[i] + t + 1).isEmpty())
            return true;


        set.add((long)nums[i]);
        if(set.size() == k+1)
            set.remove((long)nums[i - k]);
    }

    return false;

}
```

## 8. 栈和队列

### 1. 用栈实现括号匹配

[20. Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)

思路：遇到左括号入栈，右括号判断是否与当前栈顶元素匹配

```java
public boolean isValid(String s) {
    LinkedList<Character> stack = new LinkedList<>();
    for(char c : s.toCharArray())
        switch(c){
            case '(':
            case '{':
            case '[':
                stack.push(c);
                break;
            case ')':
                if(stack.isEmpty() || stack.pop() != '(')
                    return false;
                break;
            case '}':
                if(stack.isEmpty() || stack.pop() != '{')
                    return false;
                break;
            case ']':
                if(stack.isEmpty() || stack.pop() != '[')
                    return false;
                break;                    
        }

    return stack.isEmpty();
}
```

# LeetCode周赛

## 第 168 场周赛

### 1.[统计位数为偶数的数字](https://leetcode-cn.com/problems/find-numbers-with-even-number-of-digits/)（签到题）

```java
public int findNumbers(int[] nums) {

    int res = 0;

    for(int t : nums){
        int count = 0;
        while(t != 0){
            count++;
            t /= 10;
        }

        if(count % 2 == 0){
            res++;
        }
    }

    return res;

}
```

### 2.[划分数组为连续数字的集合](https://leetcode-cn.com/problems/divide-array-in-sets-of-k-consecutive-numbers/)

考虑到连续数组和存在重复元素可以使用优先队列，每次按照堆顶元素的大小依次删除，当出队失败则返回false

```java
public boolean isPossibleDivide(int[] nums, int k) {

    int len = nums.length;

    if(len % k != 0){
        return false;
    }

    PriorityQueue<Integer> queue = new PriorityQueue<>();

    for(int t : nums) {
        queue.offer(t);
    }

    while(!queue.isEmpty()){
        int start = queue.poll();
      	//从start到start + 1按元素依次从堆中删除
        for(int i = start + 1; i < start + k; i++){
            if(!queue.remove(i)){
                return false;
            }
        }

    }
    return true;
}
```

### 3.[子串的最大出现次数](https://leetcode-cn.com/problems/maximum-number-of-occurrences-of-a-substring/)（纸老虎）

读题意自然想到滑动窗口

```java
public int maxFreq(String s, int maxLetters, int minSize, int maxSize) {

    Map<String, Integer> map = new HashMap<>();
    int start = 0, end = minSize, len = s.length(), res = 0;
    while (start < len - minSize + 1) {

        int count = count(s, start, end);
        if (count > maxLetters) {
            start++;
            end++;
        } else {
            //依照maxsize扩展字符串
            while (end < Math.min(start + maxSize + 1, len + 1) && count <= maxLetters) {
                String t = s.substring(start, end++);
                map.put(t, map.getOrDefault(t, 0) + 1);
                if (end <= len)
                    count = count(s, start, end);
            }

            start++;
            end = start + minSize;
        }
    }

    for (int t : map.values()) {
        res = Math.max(t, res);
    }

    return res;
}

private int count(String s, int start, int end) {
    Set<Character> set = new HashSet<>();
    for (int i = start; i < end; i++) {
        set.add(s.charAt(i));
    }

    return set.size();
}
```

读懂题意（看完大神的题解）发现此题就是纸老虎，maxsize就是唬人的，完全不用考虑，完全不用滑动窗口，直接暴力就行了，因为如果maxsize为解，那么minsize必然也为解，改完之后时间直接变为10%

```java
public int maxFreq(String s, int maxLetters, int minSize, int maxSize) {

    Map<String, Integer> map = new HashMap<>();

    int start = 0, end = minSize, len = s.length(), res = 0;

    while (start < len - minSize + 1) {

        int count = count(s, start, end);
        if (count <= maxLetters) {
            String t = s.substring(start, end);

            map.put(t, map.getOrDefault(t, 0) + 1);
        }
        start++;
        end++;

    }

    for (int t : map.values()) {
        res = Math.max(t, res);
    }

    return res;
}

private int count(String s, int start, int end) {
    Set<Character> set = new HashSet<>();
    for (int i = start; i < end; i++) {
        set.add(s.charAt(i));
    }

    return set.size();
}
```

### 4.[你能从盒子里获得的最大糖果数](https://leetcode-cn.com/problems/maximum-candies-you-can-get-from-boxes/)

BFS+模拟，直接上代码

```java
public int maxCandies(int[] status, int[] candies, int[][] keys, int[][] containedBoxes, int[] initialBoxes) {

    int len = status.length, res = 0;
    boolean[] isVisited = new boolean[len]; //访问标志
    Set<Integer> nowKeys = new HashSet<>(), nowBoxes = new HashSet<>(); //存放现有的钥匙和现有的未打开的箱子
    Queue<Integer> queue = new LinkedList<>(); 

  	//初始化
    for(int t : initialBoxes)
        if(status[t] == 1)
            queue.offer(t);
        else
            nowBoxes.add(t);

  	
    while(!queue.isEmpty()){
        int size = queue.size();
        while(size-- > 0){

            int index = queue.poll();
            res += candies[index];

          	//遍历内含的箱子
            for(int t : containedBoxes[index])
                if((status[t] == 1 || nowKeys.contains(t)) && !isVisited[t])
                    queue.offer(t);
                else
                    nowBoxes.add(t);        

          	//遍历钥匙
            for(int t : keys[index])
                nowKeys.add(t);
            isVisited[index] = true;
        }
      	
      	//处理新增的钥匙或包
        for(int b : nowBoxes){
           if(nowKeys.contains(b) && !isVisited[b]){
                queue.offer(b);
            }
        }
    }

    return res;
}
```

## 第169场周赛

### 1.[和为零的N个唯一整数](https://leetcode-cn.com/problems/find-n-unique-integers-sum-up-to-zero/)(签到题)

```java
public int[] sumZero(int n) {
    List<Integer> res = new ArrayList<>();
    int[] r = new int[n];
    if (n % 2 == 1) {
        res.add(0);
    }
    for (int i = 1; i <= n / 2; i++) {
        res.add(i);
        res.add(-i);
    }

    for (int i = 0; i < n; i++) {
        r[i] = res.get(i);
    }
    return r;
}
```

### 2.[两棵二叉搜索树中的所有元素](https://leetcode-cn.com/problems/all-elements-in-two-binary-search-trees/)

按元素的大小在两颗树交替遍历

```java
public List<Integer> getAllElements(TreeNode root1, TreeNode root2) {

    LinkedList<TreeNode> s1 = new LinkedList<>(), s2 = new LinkedList<>();
    List<Integer> res = new ArrayList<>();

    train(root1, s1);
    train(root2, s2);

    while(!s1.isEmpty() && !s2.isEmpty()){

        TreeNode n1 = s1.peek(), n2 = s2.peek();

      	//先处理小的节点
        if(n1.val > n2.val){
            process(s2, res);
        } else {
            process(s1, res);
        }
    }

    while(!s1.isEmpty())
        process(s1, res);

    while(!s2.isEmpty())
        process(s2, res);

    return res;
}

//加入结果集
private void process(LinkedList<TreeNode> s, List<Integer> res) {
    TreeNode n = s.pop();
    res.add(n.val);
    train(n.right, s);
}

//找最小值
private void train(TreeNode root,  LinkedList<TreeNode> s){
    while(root != null){
        s.push(root);
        root = root.left;
    }
}
```

### 3.[跳跃游戏 III](https://leetcode-cn.com/problems/jump-game-iii/)

BFS

```java
public boolean canReach(int[] arr, int start) {

    int len = arr.length;
    boolean[] flag = new boolean[len];

    Queue<Integer> queue  = new LinkedList<>();

    queue.offer(start);
    flag[start] = true;

    while(!queue.isEmpty()){
        int size =  queue.size();
        while(size-- > 0){
            int index = queue.poll();

            if(arr[index] == 0){
                return true;
            }

            int t = index  + arr[index];
            if(t < len &&  !flag[t]){
                queue.offer(t);
                flag[t] = true;
            }

            t = index  - arr[index];
            if(t >= 0 &&  !flag[t]){
                queue.offer(t);
                flag[t] = true;
            }
        }

    }
    return false;

}
```

### 4.[口算难题](https://leetcode-cn.com/problems/verbal-arithmetic-puzzle/)

暴力搜索，没有用任何剪枝技巧

```java
private static final int[] POW = new int[]{1, 10, 100, 1000, 10000, 100000, 1000000};

public boolean isSolvable(String[] words, String result) {

    Set<Character> noZero = new HashSet<>(), chars = new HashSet<>();
    Map<Character, Integer> charCount = new HashMap<>();

    boolean[] isUsed = new boolean[10];
    for (String s : words) {
        int len = s.length();
        noZero.add(s.charAt(0));
        for (int i = 0; i < len; i++) {
            char c = s.charAt(i);
            chars.add(c);
            charCount.put(c, charCount.getOrDefault(c, 0) + POW[len - i - 1]);
        }
    }

    int len = result.length();
    noZero.add(result.charAt(0));
    for (int i = 0; i < len; i++) {
        char c = result.charAt(i);
        chars.add(c);
        charCount.put(c, charCount.getOrDefault(c, 0) - POW[len - i - 1]);
    }

    return backTrack(new ArrayList<>(chars), noZero, isUsed, charCount, 0, 0);
}

private boolean backTrack(List<Character> charList, Set<Character> noZero, boolean[] isUsed, Map<Character, Integer> charCount, int diff, int step) {

    if (step == charList.size())
        return diff == 0;

    char c = charList.get(step);
    for (int i = 0; i < 10; i++) {
        if ((i == 0 && noZero.contains(c)) || isUsed[i]) {

            isUsed[i] = true;
            if (backTrack(charList, noZero, isUsed, charCount, diff + charCount.get(c) * i, step + 1))
                return true;
            isUsed[i] = false;
        }
    }
    return false;

}
```



## 第170场周赛

### 1.[解码字母到整数映射](https://leetcode-cn.com/problems/decrypt-string-from-alphabet-to-integer-mapping/)（签到题）

```java
public String freqAlphabets(String s) {
  
    StringBuilder sb = new StringBuilder();
    int len = s.length();
    for (int i = 0; i < len; i++) {
        if (i + 2 < len && s.charAt(i + 2) == '#') {
            sb.append((char) ('a' + Integer.parseInt(s.substring(i, i + 2)) - 1));
            i += 2;
        } else {
            sb.append((char) ('a' + s.charAt(i) - '1'));
        }
    }
    return sb.toString();
}
```

### 2.[子数组异或查询](https://leetcode-cn.com/problems/xor-queries-of-a-subarray/)

两个相同的数异或还是原来的数，与0异或等于本身

```java
public int[] xorQueries(int[] arr, int[][] queries) {

    int len = arr.length, l = queries.length;
    int[] dp  = new int[len + 1];
    int[] res = new int[l];

    for(int i = 1; i <= len;  i++){
        dp[i] = dp[i - 1] ^ arr[i - 1];
    }

    for(int i = 0; i < l; i++){
        res[i] = dp[queries[i][0]] ^ dp[queries[i][1] + 1];
    }

    return res;
}
```

### 3.[获取你好友已观看的视频](https://leetcode-cn.com/problems/get-watched-videos-by-your-friends/)

BFS,对map排序

```java
public List<String> watchedVideosByFriends(List<List<String>> watchedVideos, int[][] friends, int id, int level) {
    int len = friends.length, n = level;
    boolean[] isVisited = new boolean[len];
    Queue<Integer> queue = new LinkedList<>();

    queue.offer(id);
    isVisited[id] = true;
    while (n-- > 0) {
        int size = queue.size();
        while (size-- > 0) {
            int index = queue.poll();

            for (int t : friends[index]) {
                if (!isVisited[t]){
                    queue.offer(t);
                    isVisited[t] = true;
                }
            }
        }
    }

    Map<String, Integer> countMap = new HashMap<>();
    while (!queue.isEmpty()) {
        int index = queue.poll();
        for (String m : watchedVideos.get(index)) {
            countMap.put(m, countMap.getOrDefault(m, 0) + 1);
        }
    }

    //这里将map.entrySet()转换成list
    List<Map.Entry<String, Integer>> list = new ArrayList<>(countMap.entrySet());
    //然后通过比较器来实现排序
    Collections.sort(list, (e1, e2) -> {
        if (e1.getValue() != e2.getValue()) {
            return e1.getValue() > e2.getValue() ? 1 : -1;
        }
        return e1.getKey().compareTo(e2.getKey());
    });

    List<String> res = new ArrayList<>();

    for (Map.Entry<String, Integer> entry : list) {
        res.add(entry.getKey());
    }

    return res;
}
```

### 4.[让字符串成为回文串的最少插入次数](https://leetcode-cn.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/)

`区间动态规划dp[i][j],表示i~j的子串成为回文串最少的插入次数`

```java
public int minInsertions(String s) {
    int len = s.length();
    int[][] dp = new int[len][len];

  
    for(int i = len - 2; i >= 0; i--){
        char c = s.charAt(i);
        for(int j = i + 1; j < len; j++){
          	//在开头或者在结尾添加字符是字符串形成回文
            dp[i][j] = Math.min(dp[i + 1][j], dp[i][j - 1]) + 1;
            if(c == s.charAt(j))
              	//这两个字符可以不考虑
                dp[i][j] = Math.min(dp[i][j], dp[i + 1][j - 1]);
        }
    }
    return dp[0][len - 1];        
}
```

## 第171场周赛

### 1.[ 将整数转换为两个无零整数的和](https://leetcode-cn.com/problems/convert-integer-to-the-sum-of-two-no-zero-integers/)（签到题）

```java
public int[] getNoZeroIntegers(int n) {
    int[] res = new int[2];
    for(int i = 1; i <= n / 2; i++){
        if(check(i) && check(n - i)){
            res[0] = i;
            res[1] = n - i;
            break;
        }
    }
    return res;        
}

private boolean check(int n){
    while(n != 0){
        if(n % 10 == 0){
            return false;
        }
        n /= 10;
    }
    return true;
}
```

### 2.[或运算的最小翻转次数](https://leetcode-cn.com/problems/minimum-flips-to-make-a-or-b-equal-to-c/)

或运算的原理

```java
public int minFlips(int a, int b, int c) {

    Set<Integer> countA = getNum(a);
    Set<Integer> countB = getNum(b);

    Set<Integer> countC = getNum(c);

    countA.removeAll(countC);
    countB.removeAll(countC);

    countC.removeAll(getNum(a));
    countC.removeAll(getNum(b));

    return countA.size() + countB.size() + countC.size();
}

private Set<Integer> getNum(int n) {
    int i = 0;
    Set<Integer> res = new HashSet<>();
    while (n != 0) {
        if (n % 2 == 1) {
            res.add(i);
        }

        i++;
        n /= 2;
    }

    return res;
}
```

### 3.[连通网络的操作次数](https://leetcode-cn.com/problems/number-of-operations-to-make-network-connected/)

可以三种方法：并查集，BFS,DFS

```java
//并查集

private int[] id;
private int[] sz;
private int count;

private void inintUF(int n){
    id = new int[n];
    sz = new int[n];
    count = n;
    Arrays.fill(sz, 1);
    for(int i = 0; i < n; i++){
        id[i] = i;
    }
}

private int find(int p){
    if(p != id[p]){
        id[p] = find(id[p]);
    }
    return id[p];
}

private void unin(int p, int q){
    int pR = find(p), qR = find(q);

    if(pR == qR)
        return;

    if(sz[pR] > sz[qR]){
        id[qR] = pR;
        sz[pR] += sz[qR];
    }else {
        id[pR] = qR;
        sz[qR] += sz[pR];
    }
    count--;
}

public int makeConnected(int n, int[][] connections) {
    int len = connections.length;
    if(len < n - 1)
        return -1;

    inintUF(n);
    for(int[] t : connections){
        unin(t[0], t[1]);
    }

    return count - 1;
}
```

```java
//BFS
public int makeConnected(int n, int[][] connections) {
    boolean[] isVisited = new boolean[n];

    Map<Integer, Set<Integer>> map = new HashMap<>();

    for(int[] temp : connections){
        for(int i = 0; i < 2; i++){
            Set<Integer> node = map.getOrDefault(temp[i], new HashSet<>());
            node.add(temp[(i + 1) % 2]);
            map.put(temp[i], node);
        }
    }
    int count = 0;
    for(int i = 0; i < n; i++){
        if(!isVisited[i]){
            count++;
            if(map.get(i) == null)
                continue;

            Queue<Integer> queue = new LinkedList<>();
            queue.add(i);
            isVisited[i] = true;
            while(!queue.isEmpty()){

                int size = queue.size();
                while(size-- > 0){
                    int index = queue.poll();
                    Set<Integer> node =  map.get(index);
                    for(int t : node){
                        if(!isVisited[t]){
                            queue.offer(t);
                            isVisited[t] = true;
                        }
                    }
                }
            }

        }

    }

    return connections.length >= n - 1 ? count - 1 : -1;
}
```

```java
//DFS
public int makeConnected(int n, int[][] connections) {
    boolean[] isVisited = new boolean[n];

    Map<Integer, Set<Integer>> map = new HashMap<>();

    for (int[] temp : connections) {
        for (int i = 0; i < 2; i++) {
            Set<Integer> node = map.getOrDefault(temp[i], new HashSet<>());
            node.add(temp[(i + 1) % 2]);
            map.put(temp[i], node);
        }
    }

    int count = 0;
    for (int i = 0; i < n; i++) {
        if (!isVisited[i]) {
            count++;

            if (map.get(i) == null)
                continue;

            dfs(isVisited, map, i);
        }
    }

    return connections.length >= n - 1 ? count - 1 : -1;

}

private void dfs(boolean[] isVisited, Map<Integer, Set<Integer>> map, int n) {

    isVisited[n] = true;
    Set<Integer> nodes = map.get(n);

    for (int node : nodes) {
        if (!isVisited[node]) {
            dfs(isVisited, map, node);
        }
    }

}
```

### 4.[二指输入的的最小距离](https://leetcode-cn.com/problems/minimum-distance-to-type-a-word-using-two-fingers/)

`动态规划dp[i][l][r]表示：第i个字母时左手和右手的位置，当前字符c用左手输入dp[i][c][r]=Math.min(dp[i][c][r], dp[i - 1][l][r]+len(c, l)),右手同理`

```java
public int minimumDistance(String word) {
    int len = word.length(), res = Integer.MAX_VALUE;

    int[][][] dp = new int[len+1][26][26];

    for(int i = 1; i <= len; i++){
        for(int j = 0; j < 26; j++){
            Arrays.fill(dp[i][j], Integer.MAX_VALUE);
        }
    }

    for(int i = 1; i <= len; i++){
        int c = word.charAt(i - 1) - 'A';
        for(int l = 0; l < 26; l++){
            for(int r = 0; r < 26; r++){
                if(dp[i - 1][l][r] != Integer.MAX_VALUE){
                    dp[i][c][r] = Math.min(dp[i][c][r], dp[i - 1][l][r] + len(l, c));
                    dp[i][l][c] = Math.min(dp[i][l][c], dp[i - 1][l][r] + len(r, c));
                }
                if(i == len){
                    res = Math.min(res, dp[i][c][r]);
                    res = Math.min(res, dp[i][l][c]);
                }                    
            }
        }
    }
    return res;
}

private int len(int a, int b){
    int x1 = a / 6, y1 = a % 6;
    int x2 = b / 6, y2 = b % 6;
    return Math.abs(x1 - x2) + Math.abs(y1 - y2);
}
```

## 第172场周赛

### 1.[6 和 9 组成的最大数字](https://leetcode-cn.com/problems/maximum-69-number/)(签到题)

选最高位一个数字从6反转为9

```java
public int maximum69Number(int num) {
    LinkedList<Integer> stack = new LinkedList<>();

    while (num != 0) {
        stack.push(num % 10);
        num /= 10;
    }
    boolean flag = true;
    int res = 0;
    while (!stack.isEmpty()) {

        int t = stack.pop();
        if (flag && t == 6) {
            t = 9;
            flag = false;
        }
        res = res * 10 + t;
    }
    return res;
}
```

### 2.[竖直打印单词](https://leetcode-cn.com/problems/print-words-vertically/)

按列生成单词将尾部空格替换

```java
public List<String> printVertically(String s) {
    String[] temp = s.split(" ");
    int len = temp.length, maxLen = 0;
    List<String> res = new ArrayList<>();
    for (String t : temp) {
        maxLen = Math.max(maxLen, t.length());
    }

    for (int i = 0; i < maxLen; i++) {
        StringBuilder sb = new StringBuilder();

        for (String t : temp) {
            if (i < t.length()) {
                sb.append(t.charAt(i));

            } else {
                sb.append(' ');
            }
        }

        for (int j = len - 1; j >= 0; j--) {
            if (sb.charAt(j) == ' ') {
                sb.deleteCharAt(j);
            } else {
                break;
            }
        }
        res.add(sb.toString());
    }
    return res;
}
```

### 3.[删除给定值的叶子节点](https://leetcode-cn.com/problems/delete-leaves-with-a-given-value/)

先处理叶子结点，再处理父节点，后序遍历，使用带返回值的递归函数直接删除叶子结点，因为删除叶子结点的时候需要父节点

```java
public TreeNode removeLeafNodes(TreeNode root, int target) {

    if(root == null)
        return root;

    root.left = removeLeafNodes(root.left, target);
    root.right = removeLeafNodes(root.right, target);

    if(root.left == null && root.right == null && root.val == target){
        return null;   
    }else{
        return root;
    }
}
```

### 4.[灌溉花园的最少水龙头数目](https://leetcode-cn.com/problems/minimum-number-of-taps-to-open-to-water-a-garden/)

`lands[i]表示覆盖i-1到i土地的水龙头所覆盖的土地的最右边的土地，land[0]=5,表示覆盖0-1这块土地的水龙头里，最远覆盖到4-5这块土地`

```java
public int minTaps(int n, int[] ranges) {

    int[] lands = new int[n];

    for (int i = 0; i < n + 1; i++) {
        int l = Math.max(i - ranges[i], 0);
        int r = Math.min(i + ranges[i], n);
        for (int j = l; j < r; j++) {
            lands[j] = Math.max(lands[j], r);
        }
    }

    int res = 0, cur = 0;


    while (cur < n){
        if(lands[cur] == 0)
            return -1;
        cur =  lands[cur];
        res++;
    }
    return res;

}
```

## 第173场周赛

### 1.[删除回文子序列](https://leetcode-cn.com/problems/remove-palindromic-subsequences/)

仔细审题，字符串只包含a和b两种字符，所以只需判断字符串是否为回文串

```java
public int removePalindromeSub(String s) {

    int len = s.length(), j = len - 1;
    if(len == 0)
        return 0;


    for(int i = 0; i < len / 2; i++){
        if(s.charAt(i) != s.charAt(j--))
            return 2;
    }

    return 1;

}
```

### 2.[餐厅过滤器](https://leetcode-cn.com/problems/filter-restaurants-by-vegan-friendly-price-and-distance/)

模拟题，按题目要求过滤,排序即可

```java
public List<Integer> filterRestaurants(int[][] restaurants, int veganFriendly, int maxPrice, int maxDistance) {
    List<int[]> t = new ArrayList<>();
    List<Integer> res = new ArrayList<>();

    for (int[] temp : restaurants) {
        if (((veganFriendly == 1 && temp[2] == 1) || (veganFriendly == 0))
                && temp[3] <= maxPrice
                && temp[4] <= maxDistance) {

            t.add(temp);

        }
    }

    t.sort((t1, t2) -> {

        if (t1[1] != t2[1]) {
            return t1[1] > t2[1] ? -1 : 1;
        }
        if (t1[0] != t2[0]) {
            return t1[0] > t2[0] ? -1 : 1;
        }

        return 0;
    });

    for (int[] temp : t) {
        res.add(temp[0]);
    }

    return res;
}
```

### 3.[阈值距离内邻居最少的城市](https://leetcode-cn.com/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/)

floyd算法求多源最短路径

```java
public int findTheCity(int n, int[][] edges, int distanceThreshold) {

    int[][] distance = new int[n][n];

    for(int[] t : distance)
        Arrays.fill(t, 100000);

    for(int[] t : edges){
        distance[t[0]][t[1]] = t[2];
        distance[t[1]][t[0]] = t[2];
    }

  //floyd
    for(int k = 0; k < n; k++){
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                if(i == j)
                    continue;
                distance[i][j] = Math.min(distance[i][k] + distance[k][j], distance[i][j]);
            }
        }
    }

    int[] counts = new int[n];
    int min = 101;
    for(int i = 0; i < n; i++){

        int cnt = 0;

        for(int t : distance[i])
            if(t <= distanceThreshold)
                cnt++;

        min = Math.min(cnt, min);
        counts[i] = cnt;
    }

    for(int i = n - 1; i >= 0; i--)
        if(counts[i] == min)
            return i;


    return 0;

}
```

### 4.[工作计划的最低难度](https://leetcode-cn.com/problems/minimum-difficulty-of-a-job-schedule/)

`dp[i][j]前i天完成前j项任务，可以选择前i-1天完成k-1项任务，最后一天完成k...j的任务，dp[i][j]=dp[i-1][j-1]+max(jobDifficulty[k]...jobDifficulty[j])`

```java
public int minDifficulty(int[] jobDifficulty, int d) {

    int cnt = jobDifficulty.length;
    int[][] dp = new int[d][cnt];

    if(d > cnt)
        return -1;

    for(int i = 0; i < d; i++)
        Arrays.fill(dp[i], Integer.MAX_VALUE / 3);

    int max = 0;
    for(int i = 0; i < cnt; i++){
        max = Math.max(jobDifficulty[i], max);
        dp[0][i] = max;            
    }


    for(int i = 1; i < d; i++){

        for(int j = i ; j < cnt; j++){
            max = jobDifficulty[j];
            for(int k = j; k >= i; k--){
                max = Math.max(jobDifficulty[k], max);
                dp[i][j] = Math.min(dp[i][j], dp[i - 1][k - 1] + max);
            }

        }
    }
    return dp[d - 1][cnt - 1];
}
```

## 第174场周赛

### 1.[方阵中战斗力最弱的 K 行](https://leetcode-cn.com/problems/the-k-weakest-rows-in-a-matrix/)

签到题，索引绑定然后排序

```java
public int[] kWeakestRows(int[][] mat, int k) {

    Map<Integer, Integer> count = new HashMap<>();
    int[] res = new int[k];
    for (int i = 0; i < mat.length; i++) {
        count.put(i, 0);
        for (int t : mat[i]) {
            if (t == 1)
                count.put(i, count.get(i) + 1);
        }
    }

    List<Map.Entry<Integer, Integer>> entryList = new ArrayList<>(count.entrySet());
    entryList.sort((e1, e2) -> {
        int v1 = e1.getValue(), v2 = e2.getValue(), k1 = e1.getKey(), k2 = e2.getKey();
        if (v1 != v2) {
            return v1 > v2 ? 1 : -1;
        }

        if (k1 != k2) {
            return k1 > k2 ? 1 : -1;
        }
        return 0;
    });

    for (int i = 0; i < k; i++) {
        res[i] = entryList.get(i).getKey();
    }
    return res;
}
```

### 2.[数组大小减半](https://leetcode-cn.com/problems/reduce-array-size-to-the-half/)

贪心算法：每次减去出现次数最多的数字

```java
public int minSetSize(int[] arr) {
    int len = arr.length, res = 0, nowLen = len;
    Map<Integer, Integer> count = new HashMap<>();
    for(int t : arr)
        count.put(t, count.getOrDefault(t, 0) + 1);

    List<Map.Entry<Integer, Integer>> entryList = new ArrayList<>(count.entrySet());

    entryList.sort((e1, e2) -> {
        int v1 = e1.getValue(), v2 = e2.getValue(), k1 = e1.getKey(), k2 = e2.getKey();
        if (v1 != v2) {
            return v1 > v2 ? -1 : 1;
        }
        return 0;
    });

    int i = 0;
    while(nowLen > len / 2){
        res++;
        nowLen -= entryList.get(i++).getValue();
    }
    return res;  
}
```

### 3.[分裂二叉树的最大乘积](https://leetcode-cn.com/problems/maximum-product-of-splitted-binary-tree/)

遍历所有可能的子树，求乘积时可以用整个数的和减去已知子树的和来求

```java
class Solution {

  private long res = 0;
  private int MOD = 1000000000 + 7;

  //求和
  public int sumTree(TreeNode root){

      if(root == null)
          return 0;

      return root.val + sumTree(root.left) + sumTree(root.right);
  }

  //后序遍历所有子树
  public int travl(TreeNode root, int sum){
      if(root == null)
          return 0;

      int left = travl(root.left,  sum);
      int right = travl(root.right, sum);
      int subSum = left + right + root.val  ;

      res = Math.max(res, (long)(sum - subSum) * (long)subSum);

      return subSum;

  }

  public int maxProduct(TreeNode root) {

      int sum = sumTree(root);
      travl(root, sum);
      return (int)(res % MOD);

  }
}
```

### 4.[跳跃游戏 V](https://leetcode-cn.com/problems/jump-game-v/)

`动态规划：首先按照高度排序从最低的位置开始跳,dp[i]表示当前位置可以到达的坐标数，j表示从i可以达到的位置dp[i]=Math.max(dp[j]+1,dp[i])`

```java
public int maxJumps(int[] arr, int d) {

    Map<Integer, Integer> map = new HashMap<>();
    int len = arr.length, max = 0;
    int[] dp = new int[len];

    for (int i = 0; i < len; i++) {
        map.put(i, arr[i]);
        dp[i] = 1;
    }

    List<Map.Entry<Integer, Integer>> temp = new ArrayList<>(map.entrySet());

    temp.sort((e1, e2) -> {
        int v1 = e1.getValue(), v2 = e2.getValue(), k1 = e1.getKey(), k2 = e2.getKey();
        if (v1 != v2) {
            return v1 > v2 ? 1 : -1;
        }

        if (k1 != k2) {
            return k1 > k2 ? 1 : -1;
        }
        return 0;
    });

    for (Map.Entry<Integer, Integer> e : temp) {
        int k = e.getKey();

        //向右跳
        int r = Math.min(k + d, len - 1);
        for (int i = k + 1; i <= r; i++) {
            if (arr[k] <= arr[i])
                break;
            dp[k] = Math.max(dp[i] + 1, dp[k]);

        }

        //向左跳
        int l = Math.max(0, k - d);
        for (int i = k - 1; i >= l; i--) {
            if (arr[k] <= arr[i])
                break;
            dp[k] = Math.max(dp[i] + 1, dp[k]);
        }

        max = Math.max(max, dp[k]);
    }
    return max;
}
```

## 第175场周赛

### 1.[检查整数及其两倍数是否存在](https://leetcode-cn.com/problems/check-if-n-and-its-double-exist/)

签到题，可以使用HashMap

```java
public boolean checkIfExist(int[] arr) {
    int len = arr.length;
    for(int i = 0; i < len; i++){
        for(int j  = 0; j < len; j++){
            if(i != j && arr[i] == 2 * arr[j])
                return true;
        }
    }

    return false;
}
```

### 2.[制造字母异位词的最小步骤数](https://leetcode-cn.com/problems/minimum-number-of-steps-to-make-two-strings-anagram/)

求两个字符串字符的差异

```java
public int minSteps(String s, String t) {

    Map<Character, Integer> sCount = new HashMap<>(), tCount = new HashMap<>();

    for (char c : s.toCharArray())
        sCount.put(c, sCount.getOrDefault(c, 0) + 1);

    for (char c : t.toCharArray())
        tCount.put(c, tCount.getOrDefault(c, 0) + 1);

    int res = 0;
    for (Map.Entry<Character, Integer> e : sCount.entrySet()) {
        int tc = tCount.getOrDefault(e.getKey(), 0);
        if (e.getValue() > tc)
            res += (e.getValue() - tc);
    }

    return res;
}
```

### 3.[推文计数](https://leetcode-cn.com/problems/tweet-counts-per-frequency/)

用TreeSet对时间进行排序

```java
class TweetCounts {

  Map<String, TreeSet<Integer>> map;

  public TweetCounts() {
      map = new HashMap<>();

  }

  public void recordTweet(String tweetName, int time) {
      TreeSet<Integer> record = map.getOrDefault(tweetName, new TreeSet<>());
      record.add(time);

      map.put(tweetName, record);

  }

  public List<Integer> getTweetCountsPerFrequency(String freq, String tweetName, int startTime, int endTime) {

      TreeSet<Integer> record = map.get(tweetName);

      List<Integer> res = new ArrayList<>();

      int start = startTime, end = endTime, deal = 0;

      switch (freq) {
          case "minute":
              deal = 60;
              break;
          case "hour":
              deal = 3600;
              break;
          case "day":
              deal = 86400;
              break;
          default:
              break;
      }
      while (start < end + 1) {
          int t = Math.min(start + deal, end + 1);
          res.add(record.subSet(start, t).size());

          start = t;
      }
      return res;
  }
}
```

### 4.[参加考试的最大学生数](https://leetcode-cn.com/problems/maximum-students-taking-exam/)

`状态压缩的动态规划dp[i][j]表示第i行为的坐位方式为j，dp[i][j]=max(dp[i-1][k]+count(j),dp[i][j])`

```java
class Solution {

  private Map<Integer, Integer> map = new HashMap<>();

  public int maxStudents(char[][] seats) {
      int m = seats.length, n = seats[0].length, max = 0;

      int[][] dp = new int[m][1 << n];

      for (int i = 0; i < m; i++) {
          for (int j = 0; j < (1 << n); j++) {
              if (!isValid(seats, i, j))
                  continue;

              if (i == 0) {
                  dp[i][j] = count(j);
              } else {
                  int cnt = count(j);
                  for (int k = 0; k < (1 << n); k++) {
                      if (((j & (k << 1)) | (j & (k >> 1))) != 0)
                          continue;

                      dp[i][j] = Math.max(dp[i][j], dp[i - 1][k] + cnt);
                  }
              }

              max = Math.max(dp[i][j], max);
          }

      }

      return max;
  }

  private boolean isValid(char[][] seats, int i, int j) {
      if (((j & (j << 1)) | (j & (j >> 1))) != 0)
          return false;

      String s = Integer.toBinaryString(j);
      int n = seats[0].length, len = s.length();
      for (int k = len - 1, t = n - 1; k >= 0; k--, t--) {
          if (s.charAt(k) == '1' && seats[i][t] == '#')
              return false;
      }

      return true;
  }

  private int count(int j) {
      if (map.get(j) == null) {
          String s = Integer.toBinaryString(j);
          int cnt = 0;
          for (char c : s.toCharArray())
              if (c == '1')
                  cnt++;

          map.put(j, cnt);
      }

      return map.get(j);
  }

}
```

## 第176场周赛

### 1.[统计有序矩阵中的负数](https://leetcode-cn.com/problems/count-negative-numbers-in-a-sorted-matrix/)

```java
public int countNegatives(int[][] grid) {
    int n = grid.length, m = grid[0].length, res = 0;
    for(int i = 0; i < n; i++){

        if(grid[i][0] < 0){
            res += (n - i) * m;
            break;
        }
        for(int j = 0; j < m; j++){
            if(grid[i][j] < 0){
                res += (m - j);
                break;
            }
        }
    }
    return res;
}
```

### 2.[最后 K 个数的乘](https://leetcode-cn.com/problems/product-of-the-last-k-numbers/)

```java
class ProductOfNumbers {
    private List<Integer> dp;
    
    public ProductOfNumbers() {
        dp = new ArrayList<>();
        dp.add(1);      
    }

    public void add(int num) {

        if(num == 0){
            dp = new ArrayList<>();
            dp.add(1);
        }else{  
            dp.add(dp.get(dp.size() - 1) * num);
        }
    }

    public int getProduct(int k) {
        
        if(k >= dp.size())
            return 0;
        
                
        return  dp.get(dp.size() - 1) / dp.get(dp.size() - k - 1) ;
    }
}
```

### 3.[最多可以参加的会议数目](https://leetcode-cn.com/problems/maximum-number-of-events-that-can-be-attended/)

贪心算法，每次参加当前可以参加会议里面最早结束的会议

```java
public int maxEvents(int[][] events) {

    TreeMap<Integer, List<Integer>> map = new TreeMap<>();

    for (int[] temp : events) {
        List<Integer> list = map.getOrDefault(temp[0], new ArrayList<>());
        list.add(temp[1]);
        map.put(temp[0], list);
    }

    PriorityQueue<Integer> queue = new PriorityQueue<>();

    int res = 0, now = 1, n = events.length;

    while (!map.isEmpty() || !queue.isEmpty()) {

        if (!map.isEmpty()) {

            if (now < map.firstKey() && queue.isEmpty())
                now = map.firstKey();

            if (now >= map.firstKey())
                queue.addAll(map.pollFirstEntry().getValue());

        }

        while (!queue.isEmpty() && queue.peek() < now)
            queue.poll();

        if (!queue.isEmpty()) {
            queue.poll();
            res++;
        }

        now++;
    }

    return res;

}
```

### 4.[多次求和构造目标数组](https://leetcode-cn.com/problems/construct-target-array-with-multiple-sums/)

反向构造从最大的数开始

```java
public boolean isPossible(int[] target) {
    Queue<Integer> queue = new PriorityQueue<>(Comparator.reverseOrder());
    long sum = 0;
    for (int t : target) {
        queue.offer(t);
        sum += t;
    }

    while (true) {
        long pSum = queue.poll();
        if (queue.isEmpty() || pSum == 1)
            return pSum == 1;

        if (queue.peek() == 1) {
            return (pSum - 1) % (sum - pSum) == 0;
        } else {
            long n = (pSum - queue.peek()) / (sum - pSum) + 1;
            long x = pSum - n * (sum - pSum);
            if (x < 1)
                return false;
            sum = pSum - (n - 1) * (sum - pSum);
            queue.offer((int) x);
        }
    }
}
```

## 第177场周赛

### 1.[日期之间隔几天](https://leetcode-cn.com/problems/number-of-days-between-two-dates/)

```java
class Solution {
    public int daysBetweenDates(String date1, String date2) {
        String[] d1 = date1.split("-");
        String[] d2 = date2.split("-");

        int[][] date = new int[2][3];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                if (i == 0)
                    date[i][j] = Integer.parseInt(d1[j]);
                else
                    date[i][j] = Integer.parseInt(d2[j]);
            }
        }

        int di1 = cnt(date[0]), di2 = cnt(date[1]);

        return Math.abs(di1 - di2);

    }

    private int cnt(int[] date) {

        int[] dic = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        int year = 1971, month = 1, day = 1, res = 0;
        while (date[0] != year || date[1] != month || date[2] != day) {
            res++;
            day++;

            if (month == 2 && isLeap(year) ) {
                if(day > dic[month] + 1){
                    day = 1;
                    month++;
                }
            } else if (day > dic[month]) {
                day = 1;
                month++;
            }

            if (month > 12) {
                month = 1;
                year++;
            }
        }

        return res;

    }

    private boolean isLeap(int year) {
        return year % 400 == 0 || (year % 4 == 0 && year % 100 != 0);
    }

}
```

### 2.[验证二叉树](https://leetcode-cn.com/problems/validate-binary-tree-nodes/)

计算节点的入度

```java
public boolean validateBinaryTreeNodes(int n, int[] leftChild, int[] rightChild) {
    int[] flag = new int[n];

    for(int i = 0; i < n; i++){
       if(leftChild[i] != -1)
           flag[leftChild[i]]++;

        if(rightChild[i] != -1)
            flag[rightChild[i]]++;
    }

    int root = 0;
    for(int t : flag){
        if(t > 1)
            return false;

        if(t == 0)
            root++;
    }

    return root == 1;
}
```

### 3.[最接近的因数](https://leetcode-cn.com/problems/closest-divisors/)

因数越接近乘积越大，所以从最大的因数开始遍历

```java
public int[] closestDivisors(int num) {
    int diff = Integer.MAX_VALUE;
    int[] res = new int[2];
    for (int temp = num + 1; temp <= num + 2; temp++)
        for (int i = new Double(Math.sqrt(temp)).intValue(); i >= 0; i--)
            if (temp % i == 0) {
                int s = temp / i;
                if (Math.abs(s - i) < diff) {
                    res[0] = i;
                    res[1] = s;
                    diff = Math.abs(s - i);
                }
                break;
            }
    return res;
}
```

### 4.[形成三的最大倍数](https://leetcode-cn.com/problems/largest-multiple-of-three/)

`sum % 3 == 0,按大小输出，sum % 3 = 1，删除1个结果为1，或者两个结果为2，sum % 3 = 2同理`

```java
class Solution {
    public String largestMultipleOfThree(int[] digits) {
        int[] cnt = new int[10];
        
        int sum = 0;
        for(int t : digits){
            cnt[t]++;
            sum += t;
        }
        
        StringBuilder sb = new StringBuilder();
        if(sum % 3 == 1){
            if(!del(cnt, 1, 1) && !del(cnt, 2, 2))
                return "";
                      
        }else if(sum % 3 == 2){
            if(!del(cnt, 2, 1) && !del(cnt, 1, 2))
                return "";
        }
        
        for(int i = 9; i >= 0; i--){
            while(cnt[i]-- > 0)
                sb.append(i);
        }
        
        if (sb.length() > 0 && sb.charAt(0) == '0')
            return "0";
        
        return sb.toString();
    }
    
    private boolean del(int[] cnt, int mod, int num){
        
        for(int i = 1; i <= 8; i++)
            while(i % 3 == mod && cnt[i] > 0 && num > 0){
                cnt[i]--;
                num--;
            }
        
        return num == 0;
        
    }
}
```

## 第178场周赛

### 1.[有多少小于当前数字的数字](https://leetcode-cn.com/problems/how-many-numbers-are-smaller-than-the-current-number/)

```java
public int[] smallerNumbersThanCurrent(int[] nums) {
    int len = nums.length;
    int[] res = new int[len], temp = nums.clone();

    Arrays.sort(temp);
    for (int i = 0; i < len; i++)
        res[i] = leftBound(temp, nums[i]);

    return res;

}

//二分查找
private int leftBound(int[] nums, int target) {
    int l = 0, r = nums.length;
    while (l < r) {
        int mid = (l + r) / 2;
        if (nums[mid] == target) {
            r = mid;
        } else if (nums[mid] > target) {
            r = mid;
        } else if (nums[mid] < target) {
            l = mid + 1;
        }
    }
    return l;
}
```

### 2.[通过投票对团队排名](https://leetcode-cn.com/problems/rank-teams-by-votes/)

简单的排序

```java
public String rankTeams(String[] votes) {

    Map<Character, int[]> map = new HashMap<>();
    int len = votes[0].length();

    for (String s : votes) {
        for (int i = 0; i < len; i++) {
            int[] t = map.getOrDefault(s.charAt(i), new int[len]);
            t[i]++;
            map.put(s.charAt(i), t);
        }
    }

    List<Map.Entry<Character, int[]>> list = new ArrayList<>(map.entrySet());

    list.sort((e1, e2) -> {
        int[] t1 = e1.getValue(), t2 = e2.getValue();

        for (int i = 0; i < len; i++) {
            if (t1[i] != t2[i])
                return t1[i] > t2[i] ? -1 : 1;

        }

        return e1.getKey().compareTo(e2.getKey());

    });

    StringBuilder sb = new StringBuilder();
    for (Map.Entry<Character, int[]> entry : list)
        sb.append(entry.getKey());

    return sb.toString();
}
```

### 3.[二叉树中的列表](https://leetcode-cn.com/problems/linked-list-in-binary-tree/)

就是寻找子二叉树

```java
public boolean isSubPath(ListNode head, TreeNode root) {
    if(head == null)
        return true;

    if(root == null)
        return head == null;


   return isSub(head, root) || isSubPath(head, root.left) || isSubPath(head, root.right);

}

//判断是否含有链表
private boolean isSub(ListNode head, TreeNode root){
    if(head == null)
        return true;

    if(root == null)
        return head == null;

    if(root.val != head.val)
        return false;

    return isSub(head.next, root.left) || isSub(head.next, root.right);        
}
```

### 4.[使网格图至少有一条有效路径的最小代价](https://leetcode-cn.com/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/)

图的最短路径，权值分别为1和0

```java
class Solution {
    private int[][] drict = new int[][]{{}, {0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    public int minCost(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        
        int[][] dit = new int[m][n];
        
        
        for(int i = 0; i < m; i++)
            Arrays.fill(dit[i], -1);
        
        dit[0][0] = 0;
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[]{0, 0, 0});
        
        while(!queue.isEmpty()){
            
            int[] temp =queue.poll();
            
            for(int i = 1; i <= 4; i++){
                int x = temp[0] + drict[i][0],
                y = temp[1] + drict[i][1];
                
                if(x >= m || x < 0 || y >= n || y < 0)
                    continue;
                
                int add = i == grid[temp[0]][temp[1]] ? 0 : 1;
                
                if(dit[x][y] == -1 || dit[x][y] > temp[2] + add){
                    
                    dit[x][y] = temp[2] + add;
                    queue.offer(new int[]{x, y, dit[x][y]});
                }
                     
            }
        }
        return Math.max(0, dit[m - 1][n - 1]);
    }
}
```

## 每日一题

### [4.9 括号生成](https://leetcode-cn.com/problems/generate-parentheses/)

暴力回溯法，加剪枝操作

```java
public List<String> generateParenthesis(int n) {
    List<String> res = new ArrayList<>();

    dfs("", n, n, res);
    return res;
}

private void dfs(String s, int l, int r, List<String> res){

    if(l == 0 && r == 0){
        res.add(s);
        return;
    }

    if(l > 0){
        dfs(s + "(", l - 1, r, res);
    }

    if(r > l){
        dfs(s + ")", l, r - 1, res);      
    }
} 
```

w[739. 每日温度](https://leetcode-cn.com/problems/daily-temperatures/)