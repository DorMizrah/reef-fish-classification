
---  

# מודל סיווג תמונות של דגי שונית באמצעות רשת קונבולוציה (CNN) ו-ResNet50

### מבוא
מטרת קוד זה היא לבנות מודל סיווג תמונות מתקדם על מנת לזהות ולסווג מינים שונים של דגי שונית. המודל משתמש בטכניקות של ראיית מכונה עמוקה (Deep Learning) ומתבסס על רשת קונבולוציה נוירונית (CNN) בשם ResNet50, שהותאמה למשימה זו באמצעות עיבוד מקדים ואימון על מאגר תמונות.

---

## הסבר על הקוד:

### 1. גישה ל-Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```
**מטרה:** פקודה זו מאפשרת ל-Colab להתחבר לתיקיות ולמשאבים המאוחסנים ב-Google Drive האישי של המשתמש.
- **תיאור פעולה:** לאחר הפקודה תתקבל בקשה לאשר גישה ל-Colab לחשבון ה-Google, וכך המערכת תוכל לגשת לתמונות ולאחסן קבצים בתיקיית הפרויקט.

### 2. ייבוא ספריות נדרשות
```python
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
```
הקוד מייבא ספריות המשמשות לצרכים שונים:
- **TensorFlow ו-Keras:** לבניית המודל ואימון רשתות נוירוניות.
- **Matplotlib ו-Seaborn:** לצורך יצירת גרפים והמחשה ויזואלית של תוצאות האימון והבדיקה.
- **NumPy:** לחישובים מתמטיים מתקדמים.
- **Pathlib:** לניהול נתיבים.
- **Sklearn:** לחישוב מטריצת הבלבול (Confusion Matrix) ודוחות הסיווג (Classification Report), המסייעים בהערכת ביצועי המודל.

### 3. הגדרת נתיב נתוני התמונות
```python
data_dir = '/content/drive/MyDrive/code_for_AI/Images'
data_dir = pathlib.Path(data_dir)
```
- **תיאור:** קטע קוד זה מגדיר את הנתיב לספריית התמונות ב-Google Drive.
- **הערה:** יש לוודא כי שם התיקייה מתאים לשם הנתונים המאוחסנים בפועל. 

### 4. טעינת נתונים והגדרת תוויות (Labels)
```python
image_paths = list(data_dir.glob('*/*.jpg'))
image_paths = [str(path) for path in image_paths]
labels = [pathlib.Path(path).parent.name for path in image_paths]
class_names = sorted(set(labels))
num_classes = len(class_names)
label_to_index = dict((name, index) for index, name in enumerate(class_names))
labels_indices = [label_to_index[label] for label in labels]
```
- **מטרה:** לקרוא את כל נתיבי הקבצים ולקבל את שמות הקטגוריות מתוך תיקיות התמונה.
- **תהליך:**
  - `glob` משמש למציאת כל קבצי התמונות בתיקיות המשנה.
  - כל תמונה מקבלת תווית על פי שם התיקיה שבה היא נמצאת (משמש כמספר זיהוי).
  - `label_to_index` יוצר מיפוי בין שמות התוויות לערכים מספריים. פעולה זו נחוצה עבור פונקציית הסיווג של המודל.

### 5. ערבוב הנתונים וחלוקה למערכי אימון, אימות ובדיקה
```python
data = list(zip(image_paths, labels_indices))
np.random.seed(123)
np.random.shuffle(data)
image_paths, labels_indices = zip(*data)
train_size = int(0.8 * total_images)
val_size = int(0.1 * total_images)

train_paths = image_paths[:train_size]
train_labels = labels_indices[:train_size]
val_paths = image_paths[train_size:train_size + val_size]
val_labels = labels_indices[train_size:train_size + val_size]
test_paths = image_paths[train_size + val_size:]
test_labels = labels_indices[train_size + val_size:]
```
- **מטרה:** לחלק את מאגר הנתונים למערכי אימון (80%), אימות (10%) ובדיקה (10%).
- **תהליך:** ערבוב המידע מתבצע כדי להבטיח חלוקה אקראית ולמנוע הטיות. הנתונים מחולקים באחוזים קבועים, מה שמאפשר הבטחת אחוזי הצלחה ואמינות התוצאות במהלך שלבי האימון והבדיקה.

### 6. עיבוד מקדים של תמונות
```python
img_height = 224
img_width = 224
batch_size = 32
def process_path(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img, label
```
- **מטרה:** לעבד את התמונות לפורמט המתאים לרשת ResNet50.
- **שלבי העיבוד:**
  - קריאת קבצי התמונות והמרתם לתמונות בעלות ערוץ RGB.
  - התאמת גודל התמונות (224x224) לפורמט בו מתבצע אימון המודל.
  - נורמליזציה של התמונה (preprocess_input) המותאמת במיוחד עבור מודל ResNet50 לשיפור יעילות האימון.

### 7. הגברת נתונים (Data Augmentation)
```python
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal_and_vertical'),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
    layers.RandomTranslation(0.2, 0.2),
])
def augment(image, label):
    image = data_augmentation(image)
    return image, label
```
- **מטרה:** להרחיב את מגוון הנתונים באמצעות טכניקות של שינוי התמונה, דבר המאפשר עמידות גבוהה יותר של המודל בפני שונות בנתוני הבדיקה.
- **תהליך:** הגברה זו כוללת היפוך אקראי, סיבוב, זום, שינוי קונטרסט ובהירות ועוד, אשר נועדו להגדיל את יכולת המודל להתמודד עם שינויים לא צפויים בנתוני הבדיקה.

### 8. בניית מודל ResNet50
```python
from tensorflow.keras.applications import ResNet50
base_model = ResNet50(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
base_model.trainable = True
N = 140
for layer in base_model.layers[:N]:
    layer.trainable = False
```
- **מטרה:** להטעין את מודל ResNet50 שנאמן מראש, תוך קפיאת השכבות הראשונות שלו ושמירת המידע שנלמד.
- **שלבי הבניה:** הגדרת שכבות המודל עם הגבלות על השכבות הראשונות (140 שכבות במודל זה) שהן מוקפאות לשמירת תכונות רלוונטיות שנתונים של מודל ResNet50, בשילוב האפשרות ללמידה מחודשת של השכבות הגבוהות יותר.

### 9. הגדרת שכבות המודל והתאמתו למספר מחלקות הסיווג
```python
inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)
```
- **מטרה:** בניית מודל המותאם לסיווג התמונה למספר מחלקות (מספר מיני הדגים).
- **מבנה המודל:** בנוסף למודל ה-ResNet50, שכבת Pooling לחישוב הממוצע של כל מאפיין ויישום Dropout כדי למנוע overfitting. בשכבת הפלט, פונקציית softmax המתאימה למשימות סיווג רב-קטגוריאליות.

### 10. קומפילציה ואימון המודל
```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)
```
- **מטרה:** קומפילציה ואימון המודל, תוך מעקב אחרי הדיוק ואיבוד בכל איטרציה.
- **הגדרות האימון:**
  - אופטימייזר Adam, הלומד עם שיעור למידה נמוך כדי לשפר את יציבות המודל.
  - פונקציית האיבוד `Sparse Categorical Crossentropy` מותאמת לתוויות בצורת מספר

ים במקום טקסט.

### 11. הערכת המודל וניתוח תוצאות
```python
test_loss, test_acc = model.evaluate(test_ds)
y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
y_pred = np.argmax(model.predict(test_ds), axis=-1)
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
print(classification_report(y_true, y_pred, target_names=class_names))
```
- **מטרה:** לבדוק את ביצועי המודל על מערך הבדיקה ולהפיק דוחות להערכת הדיוק.
- **תיאור תהליך:** ניתוח התוצאות בעזרת מטריצת בלבול ו-Class Report המספקים מדדי ביצוע כגון דיוק, אחזור ו-f1-score.

### 12. ויזואליזציה של תחזיות המודל
```python
for images, labels in test_ds.take(1):
    predictions = model.predict(images)
    pred_labels = np.argmax(predictions, axis=1)
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(f'True: {class_names[labels[i]]}\nPred: {class_names[pred_labels[i]]}')
        plt.axis('off')
    plt.show()
```
- **מטרה:** להציג את התחזיות של המודל על דוגמאות ממערך הבדיקה.
- **פירוט:** הצגת תמונות הבדיקה בצירוף התוויות האמיתיות ותוצאות הסיווג של המודל.
