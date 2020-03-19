{   
    "sfnet":{
        "embedding_dim": 10,

        "num_item_emb" : 1378,
        "num_category_emb" : 7,
        "num_cup_size_emb" : 12,
        "num_user_emb" : 47958,

        "activation": "relu", // relu or tanh
        "dropout": 0.3, 

        "num_targets": 3, // small, fit or large
    },
    "trainer": {
        "num_epochs": 10,
        "batch_size": 1024,
        "optimizer": {
          "lr": 0.001,
          "type": "adam"
        }
    }
}