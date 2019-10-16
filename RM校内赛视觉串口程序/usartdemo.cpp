#include "usart.h"

/**
 * 注意：该程序需要使用ubuntu！！从的
 * 并使用root权限打开程序才能打开串口！！
 */
void useUsart(){
    SendMessage sendmessage;
    Usart usart;
    while (true){
        // 单位为度
        sendmessage.pitch=0; //上正下负
        sendmessage.yaw=0; //逆正顺负
        usart.UsartSend(sendmessage);
    }
}
