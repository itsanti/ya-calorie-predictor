import os
from sklearn.model_selection import train_test_split

def split_dataset(df, random_state, save_dir="./data", val_size=0.1):
    train_df = df[df['split'] == 'train']
    test_df  = df[df['split'] == 'test']

    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=random_state)

    os.makedirs(save_dir, exist_ok=True)
    train_df.to_csv(os.path.join(save_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(save_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(save_dir, "test.csv"), index=False)

    print(f"Train set: {len(train_df)}")
    print(f"Val set:   {len(val_df)}")
    print(f"Test set:  {len(test_df)}")

    return train_df, val_df, test_df

def map_ingredients_to_text(dish_df, ingr_df):
    dish_df = dish_df.copy()
    id2ingr = dict(zip(ingr_df["id"], ingr_df["ingr"]))
    
    def ids_to_text(ingr_str):
        try:
            ids = [int(x.replace("ingr_", "")) for x in str(ingr_str).split(";")]
            ingredients = []
            for i in ids:
                if i in id2ingr:
                    ingredients.append(id2ingr[i])
                else:
                    ingredients.append(f"unknown_ingredient_{i}")
            return " ".join(ingredients) if ingredients else "unknown dish"
        except (ValueError, AttributeError) as e:
            return "unknown dish"
    
    dish_df["ingredients_text"] = dish_df["ingredients"].apply(ids_to_text)
    return dish_df
