# Disaster Rescue RL Environment

**Autonomous Drone Search & Rescue Simulation for Scale X Meta Hackathon**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## 📋 Overview

A realistic reinforcement learning environment for training autonomous drones to conduct search and rescue operations in post-disaster urban environments. Features:

- **3 Difficulty Levels**: Easy (5 victims), Medium (12 victims), Hard (25 victims)
- **Partial Observability**: Fog of war and limited visibility ranges
- **Dynamic Hazards**: Aftershocks, fires, unstable structures
- **Battery Management**: Limited power for extended missions
- **Multi-objective Optimization**: Rescue victims, deliver resources, avoid hazards
- **OpenAI Integration**: LLM-based action selection support
- **REST API**: FastAPI web interface for Hugging Face Spaces

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker (optional, for containerization)
- pip or conda

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/disaster-rescue-rl.git
cd RL_Model

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --only-binary :all:
