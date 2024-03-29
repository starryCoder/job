## 系统启动过程

1. 内核引导：BIOS开机自检之后，按照BIOS中设置的启动设备来启动，操作系统接管硬件后首先读入/boot目录下的内核文件

2. 运行init：运行init程序，init程序读取配置文件/etc/inittab

   init程序由init线程执行，pid(进程号)为1，ppid(父进程号)为0，由0号进程通过kernel_thread创建，下图为centos8的进程结构，systemd是Linux中最新的init程序

   ![linux-init](https://cdn.jsdelivr.net/gh/starryCoder/job@master/pic/linux-init.2kfsxudwh7q0.png)

   Linux下有三个特殊的进程idle进程（PID=0），init进程（PID=1），和kthreadd（PID=2）

   - **idle进程由系统自动创建，运行在内核态**，是系统创建的第一个进程，是静态创建，是唯一一个不是通过看kernel_thread或者fork创建的线程
   - **kthreadd进程由idle通过kernel_thread创建，并始终运行在内核空间，负责所有内核进程的调度和管理。**因此所有内核线程都间接或直接以kthreadd为父线程
   - **init进程由idle通过kernel_thread创建，在内核空间完成初始化后，加载init程序**。因此该进程是系统中其它所有用户进程的祖先进程，Linux启动时，在用户空间启动完init线程后，然后在启动其它系统进程，系统启动完成后，init变成守护进程监视系统其它进程。

3. 系统初始化

   许多程序需要开机启动，init进程的一大任务就是，运行这些开机启动的程序，或者叫**守护进程**。但是，不同的场景下需要运行不一样的程序，所以Linux定义了运行级别，根据运行级别可以确定需要运行哪些程序，各个运行级别的场景如下表所示

   | 运行级别 |                             定义                             |
   | :------: | :----------------------------------------------------------: |
   |    0     | 停机，关机，系统默认运行级别不能设置为0，否则系统无法正常启动 |
   |    1     | 单用户，无网络连接，不运行守护进程，不允许非超级用户登录，用于系统维护 |
   |    2     |              多用户，无网络连接，不运行守护进程              |
   |    3     |       多用户，正常启动系统，server版本系统默认运行级别       |
   |    4     |                   系统未使用，可用户自定义                   |
   |    5     |                     多用户，进入GUI模式                      |
   |    6     |     重启，默认运行级别不能设置为此级别，否则系统无法启动     |

## Linux关机

正确的关机流程：`sync>shutdown`

`shutdown -h now` 立马关机 `reboot` 重启 `halt` 停用CPU，但是仍然保持通电，系统处于低层维护状态。

总之，无论关机还是重启，首先要运行sync命令把内存的数据写到磁盘中

## Linux系统目录结构

![image](https://cdn.jsdelivr.net/gh/starryCoder/job@master/pic/image.6ko0a24danc0.png)

以下是对各个目录的解释

* **/bin**:bin是Binary的缩写，这个目录存放着最经常使用的命令
* **/boot**:存放的是启动Linux时核心文件，包括连接文件以及镜像文件
* **/dev**:dev是device的缩写，存放的是Linux的外部设备，Linux中一切皆文件
* **/etc**:Etcetera的缩写，存放系统管理所需的配置文件和子目录
* **/home**:用户的主目录
* **/lib,/lib64**:存放着系统最基本的动态连接共享库(lib64存放的为64位的链接库)，其作用类似于Windows里的DLL文件。几乎所有的应用程序都需要用到这些共享库。
* **/lost+found**:这个目录一般情况下是空的，当系统非法关机后，这里就存放了一些文件。
* **/media**:linux会自动识别的一些设备（如：光盘，U盘等），当系统识别后，会自动把设备挂载到这个目录
* **/mnt**:为了让用户临时挂载别的文件系统
* **/opt**:optional（可选的）缩写，默认是空的，是给主机安装额外软件所摆放的目录（如：MySQL，Nginx）
* **/proc**:Processes（进程）的缩写，虚拟目录是系统内存的映射，可以通过访问该目录获取系统信息，此目录的内容不在硬盘上而是存在内存里
* **/root**:系统管理员的用户目录
* **/sbin**:系统管理员使用的系统管理程序
* **/srv**:存放服务启动之后需要提取的数据
* **/sys**:该目录下安装了 2.6 内核中新出现的一个文件系统 sysfs 。sysfs 文件系统集成了下面3种文件系统的信息：针对进程信息的 proc 文件系统、针对设备的 devfs 文件系统以及针对伪终端的 devpts 文件系统。该文件系统是内核设备树的一个直观反映。当一个内核对象被创建的时候，对应的文件和目录也在内核对象子系统中被创建。
* **/tmp**:存放临时文件
* **/usr**:用户很多应用程序和文件都放在此目录下
* **/var**:这个目录存放着不断在扩充的东西（如：日志文件）
* **/run**:是一个临时文件系统，存储系统启动以来的信息。当系统重启时，这个目录下的文件应该被删掉或清除。

重要的几个目录，平时不要误删或者随意更改内部文件：`/etc`:系统中的配置文件，可能会导致系统不能启动；`/bin,/sbin`:系统预设执行文件的放置目录（如:ls,cd等）；`/var`:系统的日志文件

## Linux文件基本属性



