# Script : DNN_launch.sh
#
# Script bash invitant l'utilisateur à exécuter interface.py en utilisant ou non la carte graphique
#
# Paul Guelorget
#
echo
echo "Welcome. First and foremost, you must choose a device among CPU and GPU."
echo

YourChoice()
{
    local qst def rep
    qst="${1:-'CPU or GPU? '}"
    def="$2"
    while :
    do
        read -p "$qst" rep || exit 1
        case "$(echo "${rep:-$def}" | tr '[a-z]' '[A-Z]')" in
            CPU|C) return 0 ;;
            GPU|G) return 1 ;;
            "" )    :       ;;
            *)     echo "Wrong input: $rep" ;;
        esac
    done
}

if YourChoice "  Your choice? " 0
    then sudo THEANO_FLAGS=device=cpu python interface.py
    else sudo THEANO_FLAGS=device=gpu python interface.py
fi

