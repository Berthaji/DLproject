from train_test_all import train_and_test_CNN, train_and_test_resnet


#   Se resnet dà problemi col certificato SSL, esegui da terminale l'equivalente Windows di:
#    /Applications/Python\ 3.12/Install\ Certificates.command
#    Modifica la versione di python con la tua nella riga da eseguire nel terminale.
#    Se stai usando pyenv esegui: pip --upgrade certifi



if __name__ == '__main__':


    # train_and_test_CNN(
    #     train_dir="FER2013_binario/train",         # Percorso della directory di training
    #     test_dir="FER2013_binario/test",           # Percorso della directory di test
    #     num_epochs=10,                     # Numero di epoche di training (Non oltre 10, overfitta)
    #     batch_size=64,                     # Dimensione del batch
    #     model_save_path="models/CNN_binario_focal.pth",  # Percorso per salvare il modello
    #     test_results_path="results/CNN/test_binario_focal.csv", # cambiare nomi in base alla loss usata
    #     validation_results_path="results/CNN/validation_binario_focal.csv",
    #     mode="binary",  #binary o multiclass
    #     loss_function="focal", #"crossentropy" per cross-entropy loss, "focal" per focal loss
    #     device="mps", # mps, cuda o cpu
    #     validate=False
    # )

    # train_and_test_CNN(
    #     train_dir="FER2013_binario/train",         # Percorso della directory di training
    #     test_dir="FER2013_binario/test",           # Percorso della directory di test
    #     num_epochs=10,                     # Numero di epoche di training (Non oltre 10, overfitta)
    #     batch_size=64,                     # Dimensione del batch
    #     model_save_path="models/CNN_binario_crossentropy.pth",  # Percorso per salvare il modello
    #     test_results_path="results/CNN/test_binario_crossentropy.csv", # cambiare nomi in base alla loss usata
    #     validation_results_path="results/CNN/validation_binario_crossentropy.csv",
    #     mode="binary",  #binary o multiclass
    #     loss_function="crossentropy", #"crossentropy" per cross-entropy loss, "focal" per focal loss
    #     device="mps", # mps, cuda o cpu
    #     validate=False
    # )

    # train_and_test_CNN(
    #     train_dir="FER2013_multiclasse/train",         # Percorso della directory di training
    #     test_dir="FER2013_multiclasse/test",           # Percorso della directory di test
    #     num_epochs=10,                     # Numero di epoche di training
    #     batch_size=64,                     # Dimensione del batch
    #     model_save_path="models/CNN_multiclasse_focal.pth",  # Percorso per salvare il modello
    #     test_results_path="results/CNN/test_multiclasse_focal.csv",
    #     validation_results_path="results/CNN/validation_multiclasse_focal.csv",
    #     mode="multiclass",
    #     loss_function="focal",
    #     device="mps",
    #     validate=False
    # )

    train_and_test_CNN(
        train_dir="FER2013_multiclasse/train",         # Percorso della directory di training
        test_dir="FER2013_multiclasse/test",           # Percorso della directory di test
        num_epochs=10,                     # Numero di epoche di training
        batch_size=64,                     # Dimensione del batch
        model_save_path="models/CNN_multiclasse_crossentropy.pth",  # Percorso per salvare il modello
        test_results_path="results/CNN/test_multiclasse_crossentropy.csv",
        validation_results_path="results/CNN/validation_multiclasse_crossentropy.csv",
        mode="multiclass",
        loss_function="crossentropy",
        device="mps",
        validate=False
    )

    # train_and_test_resnet(
    #         train_dir="FER2013_binario/train",
    #         test_dir="FER2013_binario/test",
    #         num_epochs=5,
    #         batch_size=64,
    #         model_save_path="models/resnet_binario_focal.pth",
    #         validation_results_path="results/resnet/validation_binario_focal.csv",
    #         test_results_path="results/resnet/test_binario_focal.csv",
    #         mode="binary",
    #         loss_function="focal",
    #         device="mps",
    #         validate=False
    #     )
    
    # train_and_test_resnet(
    #         train_dir="FER2013_binario/train",
    #         test_dir="FER2013_binario/test",
    #         num_epochs=5,
    #         batch_size=64,
    #         model_save_path="models/resnet_binario_crossentropy.pth",
    #         validation_results_path="results/resnet/validation_binario_crossentropy.csv",
    #         test_results_path="results/resnet/test_binario_crossentropy.csv",
    #         mode="binary",
    #         loss_function="crossentropy",
    #         device="mps",
    #         validate=False
    #     )
    
    train_and_test_resnet(
            train_dir="FER2013_multiclasse/train",
            test_dir="FER2013_multiclasse/test",
            num_epochs=5,
            batch_size=64,
            model_save_path="models/resnet_multiclasse_crossentropy.pth",
            validation_results_path="results/resnet/validation_multiclasse_crossentropy.csv",
            test_results_path="results/resnet/test_multiclasse_crossentropy.csv",
            mode="multiclass",
            loss_function="crossentropy",
            device="mps",
            validate=False
        )
    
    train_and_test_resnet(
            train_dir="FER2013_multiclasse/train",
            test_dir="FER2013_multiclasse/test",
            num_epochs=5,
            batch_size=64,
            model_save_path="models/resnet_multiclasse_focal.pth",
            validation_results_path="results/resnet/validation_multiclasse_focal.csv",
            test_results_path="results/resnet/test_multiclasse_focal.csv",
            mode="multiclass",
            loss_function="focal",
            device="mps",
            validate=False
        )
    

    # # test resnet su 10 epoche di addestramento

    # train_and_test_resnet(
    #         train_dir="FER2013_binario/train",
    #         test_dir="FER2013_binario/test",
    #         num_epochs=10,
    #         batch_size=64,
    #         model_save_path="models/resnet10epoche_binario_focal.pth",
    #         validation_results_path="results/resnet_10epoche/validation_binario_focal.csv",
    #         test_results_path="results/resnet_10epoche/test_binario_focal.csv",
    #         mode="binary",
    #         loss_function="focal",
    #         device="mps",
    #         validate=False
    #     )
    
    # train_and_test_resnet(
    #         train_dir="FER2013_binario/train",
    #         test_dir="FER2013_binario/test",
    #         num_epochs=10,
    #         batch_size=64,
    #         model_save_path="models/resnet10epoche_binario_crossentropy.pth",
    #         validation_results_path="results/resnet_10epoche/validation_binario_crossentropy.csv",
    #         test_results_path="results/resnet_10epoche/test_binario_crossentropy.csv",
    #         mode="binary",
    #         loss_function="crossentropy",
    #         device="mps",
    #         validate=False
    #     )
    
    train_and_test_resnet(
            train_dir="FER2013_multiclasse/train",
            test_dir="FER2013_multiclasse/test",
            num_epochs=10,
            batch_size=64,
            model_save_path="models/resnet10epoche_multiclasse_crossentropy.pth",
            validation_results_path="results/resnet_10epoche/validation_multiclasse_crossentropy.csv",
            test_results_path="results/resnet_10epoche/test_multiclasse_crossentropy.csv",
            mode="multiclass",
            loss_function="crossentropy",
            device="mps",
            validate=False
        )
    
    train_and_test_resnet(
            train_dir="FER2013_multiclasse/train",
            test_dir="FER2013_multiclasse/test",
            num_epochs=10,
            batch_size=64,
            model_save_path="models/resnet10epoche_multiclasse_focal.pth",
            validation_results_path="results/resnet_10epoche/validation_multiclasse_focal.csv",
            test_results_path="results/resnet_10epoche/test_multiclasse_focal.csv",
            mode="multiclass",
            loss_function="focal",
            device="mps",
            validate=False
        )