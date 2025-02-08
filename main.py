from train_test_all import train_and_test_CNN, train_and_test_TL


#   Se resnet dà problemi col certificato SSL, esegui da terminale l'equivalente Windows di:
#    /Applications/Python\ 3.12/Install\ Certificates.command
#    Modifica la versione di python con la tua nella riga da eseguire nel terminale.
#    Se stai usando pyenv esegui: pip --upgrade certifi



if __name__ == '__main__':

    
    # train_and_test_TL(
    #         train_dir="FER2013_binario/train",
    #         test_dir="FER2013_binario/test",
    #         num_epochs=5,
    #         batch_size=32,
    #         model_save_path="models/efficientnet_binario_crossentropy.pth",
    #         validation_results_path="results/efficientnet/validation_binario_crossentropy.csv",
    #         test_results_path="results/efficientnet/test_binario_crossentropy.csv",
    #         mode="binary",
    #         loss_function="crossentropy",
    #         device="mps",
    #         validate=True,
    #         model_name="efficientnet"
    #     )
    
    # train_and_test_TL(
    #     train_dir="FER2013_binario/train",
    #     test_dir="FER2013_binario/test",
    #     num_epochs=5,
    #     batch_size=32,
    #     model_save_path="models/mobilenet_binario_crossentropy.pth",
    #     validation_results_path="results/mobilenet/validation_binario_crossentropy.csv",
    #     test_results_path="results/mobilenet/test_binario_crossentropy.csv",
    #     mode="binary",
    #     loss_function="crossentropy",
    #     device="mps",
    #     validate=True,
    #     model_name="mobilenet"
    # )
    

    # train_and_test_CNN(
    #     train_dir="FER2013_binario/train",
    #     test_dir="FER2013_binario/test",
    #     num_epochs=5,
    #     batch_size=32,
    #     model_save_path="models/CNN2conv_binario_crossentropy.pth",
    #     validation_results_path="results/CNN_2conv/validation_binario_crossentropy.csv",
    #     test_results_path="results/CNN_2conv/test_binario_crossentropy.csv",
    #     mode="binary",
    #     loss_function="crossentropy",
    #     device="mps",
    #     validate=True,
    #     model_name="2conv"
    # )

    train_and_test_CNN(
        train_dir="FER2013_binario/train",
        test_dir="FER2013_binario/test",
        num_epochs=5,
        batch_size=32,
        model_save_path="models/CNN4conv_binario_crossentropy.pth",
        validation_results_path="results/CNN_4conv/validation_binario_crossentropy.csv",
        test_results_path="results/CNN_4conv/test_binario_crossentropy.csv",
        mode="binary",
        loss_function="crossentropy",
        device="mps",
        validate=True,
        model_name="4conv"
    )