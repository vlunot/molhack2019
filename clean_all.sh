echo "Cleaning will delete all computed files. Are you sure?"
read -p "Type DELETE in capital letters to confirm: " -r
if [[ $REPLY == "DELETE" ]]
then
    rm preprocessed/*
    rm models/*
    rm tmp/*
    rm res/*
fi