# AI课程Git操作学习笔记

---

## 一、PR提交流程

### 1.1 复刻（Fork）目标仓库
通过浏览器打开目标GitHub仓库的网页，在页面右上角找到并点击 `Fork` 按钮，该操作会在你的GitHub账号下创建一个与原仓库完全独立的副本。

最终生成的副本仓库地址格式为：
```
https://github.com/[GitHub用户名]/[原仓库名称]
```

---

### 1.2 克隆（Clone）副本仓库到本地
在需要存放仓库的文件夹空白处右键，选择「Git Bash Here」打开命令行终端，执行以下克隆命令，将远程副本仓库下载到本地：

```bash
git clone [你Fork得到的副本仓库完整网页链接]
```

执行完成后，本地会生成一个与远程仓库内容一致的文件夹。

---

### 1.3 本地文件修改
在克隆得到的本地文件夹中，根据需求对文件进行新增、删除或修改操作（此步骤仅修改工作区文件，未纳入版本控制）。

---

### 1.4 本地仓库提交修改
再次打开「Git Bash Here」终端，依次执行以下命令，将工作区的修改先加入暂存区，再永久保存到本地仓库：

```bash
git add .                  # 将所有修改的文件加入暂存区
git commit -m "本次提交的说明信息（如：新增AI课程第3章笔记）"  # 提交到本地仓库并添加备注
```

---

### 1.5 推送（Push）修改到远程副本仓库
在终端中执行以下命令，将本地仓库的提交推送到你的GitHub远程副本仓库：

```bash
git push origin main       # 推送到main分支（若分支名不同需对应修改）
```

---

### 1.6 创建Pull Request（PR）
1. 打开浏览器，访问你的副本仓库网页（地址格式：`https://github.com/[你的用户名]/[原仓库名]`）
2. 点击页面中的「Pull requests」选项卡，再点击「New pull request」按钮
3. 配置对比分支：
   - base repository：选择老师的仓库（`chenziyang110/main`）
   - compare repository：选择你的仓库（`yzc2453/main`）
4. 填写PR标题（需清晰描述本次提交的内容）
5. 点击「Create pull request」完成PR创建

---

## 二、更新已提交的PR

### 2.1 前置准备：分支创建与切换
在本地仓库的 `main` 分支下，可通过以下命令创建并切换到新分支（推荐为不同修改创建独立分支）：

```bash
# 方式1：分步创建+切换
git branch [自定义分支名称]   # 创建新分支
git checkout [自定义分支名称] # 切换到新分支

# 方式2：一步创建并切换（推荐）
git checkout -b [你的分支名称]
```

---

### 2.2 核心步骤：同步上游仓库（拉取老师仓库最新内容）

#### 第一步：配置上游仓库（仅首次执行）
```bash
# 进入本地仓库根目录（示例路径）
cd ~/Desktop/HZU-Jiangxia-AI-Class-2026

# 查看当前已配置的远程仓库（初始仅显示origin，即你的副本仓库）
git remote -v

# 添加老师的仓库作为上游仓库（仅首次执行）
git remote add upstream https://github.com/chenziyang110/HZU-Jiangxia-AI-Class-2026.git

# 再次查看，确认upstream已添加
git remote -v
```

#### 第二步：拉取并合并上游仓库的最新更新
```bash
# 1. 切换回本地main分支
git checkout main

# 2. 从上游仓库拉取最新代码（仅下载，不合并）
git fetch upstream

# 3. 将上游仓库的main分支合并到本地main分支
git merge upstream/main

# 4. 将更新后的本地main分支推送到你的远程副本仓库
git push origin main
```

#### 第三步：在自定义分支中同步更新
```bash
# 1. 切换到需要同步的目标分支
git checkout [目标分支名称]

# 2. 将main分支的最新内容（已同步上游）合并到当前分支
git merge main

# 3. 若出现冲突，需手动编辑文件解决冲突，或使用工具：git mergetool
# 4. 解决冲突后，将更新推送到你的远程分支
git push origin [目标分支名称]
```

---

### 2.3 分支模式下提交PR的补充说明

#### 1. 创建功能分支
```bash
# 进入项目根目录
cd HZU-Jiangxia-AI-Class-2026

# 创建有语义的分支名（示例）
git checkout -b add-notes-20250121
# 或更个性化的命名
git checkout -b yangzhongci/update-notes
```

#### 2. 在分支中修改文件
根据需求在当前分支下增删改文件，操作与 `main` 分支一致。

#### 3. 提交并推送分支修改
```bash
# 确保回到项目根目录（若路径不对，执行 cd HZU-Jiangxia-AI-Class-2026）
# 将修改加入暂存区
git add .
# 提交到本地分支
git commit -m "本次修改的详细说明"
# 推送到远程对应分支（注意：不是main分支！）
git push origin [目标分支名称]
```
```

---
