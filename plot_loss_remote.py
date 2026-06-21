import json
import time
import matplotlib
# 關鍵：告訴 matplotlib 我們沒有顯示器，請使用 'Agg' 模式來把圖畫在記憶體中
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 設定檔案路徑
LOG_FILE_PATH = 'runs/train_100k/train.jsonl'
OUTPUT_IMAGE_PATH = 'training_loss_realtime.png'  # 輸出的圖片名稱

steps = []
losses = []
last_file_position = 0

print(f"開始監控日誌... 圖片將持續更新至: {OUTPUT_IMAGE_PATH}")
print("按下 Ctrl+C 可以停止腳本。")

try:
    while True:
        updated = False
        try:
            with open(LOG_FILE_PATH, 'r') as f:
                f.seek(last_file_position)
                new_lines = f.readlines()
                last_file_position = f.tell()
                
                for line_str in new_lines:
                    line_str = line_str.strip()
                    if not line_str:
                        continue
                    
                    try:
                        data = json.loads(line_str)
                        if data.get('event') == 'step' and data.get('loss') is not None:
                            steps.append(data['step'])
                            losses.append(data['loss'])
                            updated = True
                    except json.JSONDecodeError:
                        pass
                        
        except FileNotFoundError:
            print(f"等待檔案 {LOG_FILE_PATH} 建立中...", end='\r')
        
        # 如果有讀取到新資料，就重新畫一張圖並存檔
        if updated and steps:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(steps, losses, '-', color='#1f77b4', linewidth=2, label='Training Loss')
            
            ax.set_title('Real-time Training Loss', fontsize=14, fontweight='bold')
            ax.set_xlabel('Step', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            # 儲存圖片
            plt.tight_layout()
            plt.savefig(OUTPUT_IMAGE_PATH, dpi=100)
            
            # 關閉畫布以釋放記憶體 (這在無窮迴圈中非常重要！)
            plt.close(fig)
            print(f"[{time.strftime('%H:%M:%S')}] 已更新圖表至 {OUTPUT_IMAGE_PATH} (目前 Step: {steps[-1]})")
        
        # 暫停 5 秒後再檢查一次日誌 (可依需求調整更新頻率)
        time.sleep(5)

except KeyboardInterrupt:
    print("\n監控已結束。")
