# Module 01 速查表

## 镜像命令
```bash
docker pull <image>:<tag>       # 拉取镜像
docker images                   # 列出本地镜像
docker rmi <image>              # 删除镜像
docker build -t <name>:<tag> .  # 构建镜像
docker image prune              # 清理悬空镜像
```

## 容器命令
```bash
docker run <image>                          # 运行容器
docker run -it <image> bash                 # 交互模式
docker run -d --name <name> <image>         # 后台运行
docker run --rm <image>                     # 运行完自动删除
docker run -p 8888:8888 <image>             # 端口映射
docker run -v /host/path:/container/path    # 挂载卷

docker ps                   # 查看运行中容器
docker ps -a                # 查看所有容器
docker stop <container>     # 停止容器
docker rm <container>       # 删除容器
docker exec -it <container> bash  # 进入运行中容器
docker logs -f <container>  # 跟踪日志
```

## 常用 Dockerfile 指令
```dockerfile
FROM <image>:<tag>          # 基础镜像
WORKDIR /path               # 设置工作目录
COPY src dest               # 复制文件
RUN command                 # 执行命令（构建时）
CMD ["cmd", "arg"]          # 容器启动命令
ENV KEY=VALUE               # 设置环境变量
EXPOSE 8080                 # 声明端口（文档用途）
ARG BUILD_VAR               # 构建时参数
```

## ML 场景常用组合
```bash
# 挂载数据+代码，运行训练
docker run --rm \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/outputs:/workspace/outputs \
  my-train:latest python train.py

# 进入容器调试
docker run -it --rm \
  -v $(pwd):/workspace \
  my-train:latest bash
```
