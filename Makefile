# =============================================================================
# Makefile - ROCKET-2 构建与运行
# =============================================================================
# Mac 开发:
#   make dev-build    # 构建开发镜像 (x86 模拟)
#   make dev-run      # 运行开发容器
#   make dev          # 构建 + 运行
#
# 服务器生产:
#   make server-build # 构建生产镜像
#   make server-run   # 运行生产容器 (需要 NVIDIA GPU)
#   make server       # 构建 + 运行
# =============================================================================

.PHONY: dev-build dev-run dev server-build server-run server clean help

# ---------------------------------------------------------------------------
# 开发环境 (Apple Silicon Mac, x86 模拟, CPU)
# ---------------------------------------------------------------------------

dev-build:
	docker build --platform linux/amd64 -f Dockerfile.dev -t rocket2-dev .

dev-run:
	docker run -it --rm \
		--platform linux/amd64 \
		-p 7860:7860 \
		-e ROCKET_DEVICE=cpu \
		-e CUDA_VISIBLE_DEVICES= \
		-v $(PWD)/launch.py:/app/ROCKET-2/launch.py \
		-v $(PWD)/model.py:/app/ROCKET-2/model.py \
		-v $(PWD)/cfg_wrapper.py:/app/ROCKET-2/cfg_wrapper.py \
		-v $(PWD)/draw_action.py:/app/ROCKET-2/draw_action.py \
		-v $(PWD)/train.py:/app/ROCKET-2/train.py \
		-v $(PWD)/cross_view_dataset.py:/app/ROCKET-2/cross_view_dataset.py \
		-v $(PWD)/config.yaml:/app/ROCKET-2/config.yaml \
		-v $(PWD)/config_xl.yaml:/app/ROCKET-2/config_xl.yaml \
		-v $(PWD)/env_conf:/app/ROCKET-2/env_conf \
		-v $(PWD)/gallery:/app/ROCKET-2/gallery \
		-v $(PWD)/keyboard-assets:/app/ROCKET-2/keyboard-assets \
		-v $(PWD)/theme.json:/app/ROCKET-2/theme.json \
		rocket2-dev

dev: dev-build dev-run

# 交互式进入开发容器 (调试用)
dev-shell:
	docker run -it --rm \
		--platform linux/amd64 \
		-p 7860:7860 \
		-e ROCKET_DEVICE=cpu \
		-e CUDA_VISIBLE_DEVICES= \
		-v $(PWD):/app/ROCKET-2/src \
		--entrypoint /bin/bash \
		rocket2-dev

# ---------------------------------------------------------------------------
# 生产环境 (Linux 服务器, NVIDIA GPU)
# ---------------------------------------------------------------------------

server-build:
	docker build -t rocket2 .

server-run:
	docker run -it --rm \
		-p 7860:7860 \
		--gpus all \
		rocket2

server: server-build server-run

# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

clean:
	docker rmi rocket2-dev rocket2 2>/dev/null || true
	docker system prune -f

help:
	@echo ""
	@echo "  ROCKET-2 Docker 构建工具"
	@echo "  ========================"
	@echo ""
	@echo "  Mac 开发 (Apple Silicon + x86 模拟):"
	@echo "    make dev-build     构建开发镜像"
	@echo "    make dev-run       运行开发容器 (CPU 模式)"
	@echo "    make dev           构建 + 运行"
	@echo "    make dev-shell     交互式 shell (调试)"
	@echo ""
	@echo "  服务器 (Linux + NVIDIA GPU):"
	@echo "    make server-build  构建生产镜像"
	@echo "    make server-run    运行生产容器 (GPU 模式)"
	@echo "    make server        构建 + 运行"
	@echo ""
	@echo "  工具:"
	@echo "    make clean         清理 Docker 镜像"
	@echo "    make help          显示帮助"
	@echo ""
