import cv2
import albumentations as A
import matplotlib.pyplot as plt

def apply_heavy_augmentation(image_path):
    # بارگذاری تصویر
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # تعریف ترنسفورم‌های سنگین (شبیه‌سازی شرایط سخت ترافیکی)
    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.5), # تغییر نور
        A.GaussianBlur(p=0.3),            # تار کردن (حرکت سریع)
        A.RandomRain(p=0.5),              # شبیه‌سازی باران (بسیار مهم برای ITS)
        A.HorizontalFlip(p=0.5),          # قرینه‌سازی
    ])

    print("📸 Applying AI Augmentation to simulate real-world traffic...")
    
    # تولید ۹ نسخه متفاوت
    plt.figure(figsize=(12, 12))
    for i in range(9):
        augmented = transform(image=image)['image']
        plt.subplot(3, 3, i+1)
        plt.imshow(augmented)
        plt.axis('off')
        plt.title(f"Augmented {i+1}")
    
    plt.savefig('augmentation_results.png')
    plt.show()

if __name__ == "__main__":
    # از همان عکس اتوبوس دیشب یا هر عکس آمبولانسی استفاده کن
    apply_heavy_augmentation('test_traffic.jpg')