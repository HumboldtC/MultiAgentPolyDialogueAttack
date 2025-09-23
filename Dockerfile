FROM continuumio/miniconda3

# 设置工作目录
WORKDIR /app

# 设置 http/https 代理
ENV http_proxy=http://192.168.119.224:50505
ENV https_proxy=http://192.168.119.224:50505
ENV HTTP_PROXY=http://192.168.119.224:50505
ENV HTTPS_PROXY=http://192.168.119.224:50505

# 复制环境文件
COPY environment.yml .
COPY requirements.txt .
COPY ./vllm-0.8.5+cu121-cp38-abi3-manylinux1_x86_64.whl .

# 创建 conda 环境并安装依赖
RUN conda create -n mas python=3.11 -y \
    && /bin/bash -c "source activate mas \
    && pip install --upgrade pip \
    && pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124 \
    && pip install ./vllm-0.8.5+cu121-cp38-abi3-manylinux1_x86_64.whl"

# 以后所有命令都默认在 mas 环境下执行
SHELL ["conda", "run", "--no-capture-output", "-n", "mas", "/bin/bash", "-c"]

# 拷贝代码文件夹
COPY . ./MultiAgentPolyDialogueAttack
