import tkinter as tk
import tkinter.ttk as ttk
import sqlite3
import re

###############メインウィンドウ############
def root_window():

    #データ追加ボタンのアクション
    def add_data_button_click():
        root.destroy()
        input_window()

    #データ変更ボタンのアクション
    def change_data_button_click():
        root.destroy()
        change_window()

    #データ削除ボタンのアクション
    def delete_data_button_click():
        root.destroy()
        delete_window()

    #データ一覧ウィンドウ
    def list_click_window():
        leaf = tk.Tk()
        leaf.title("データ一覧")
        leaf.geometry("500x500")

        tree = ttk.Treeview(leaf,padding=10)
        tree["columns"] = (1,2,3)
        tree["show"] = "headings"
        tree.column(1,width=50)
        tree.column(2,width=100)
        tree.column(3,width=100)
        tree.heading(1,text="商品ID")
        tree.heading(2,text="商品名")
        tree.heading(3,text="金額")

        get_rec = 'select rowid, * from item'
        conn3 = sqlite3.connect('detail.sqlite3')
        for i in conn3.execute(get_rec):
            tree.insert("", "end", values=(i[0], i[1], i[2]))
        tree.pack(fill="x", padx=20)
        
        leaf.mainloop()

    #終了ボタンのアクション
    def quit_button_click():
        root.destroy()

    #メインウィンドウの設定
    root = tk.Tk()
    root.title("データ管理")
    root.geometry("500x300")
    
    #データ追加ボタンの設定
    add_data_button = tk.Button(root, text="データ追加", command=add_data_button_click)
    add_data_button.pack(fill='x', padx=20, pady=20)

    #データ変更ボタンの設定
    change_data_button = tk.Button(root, text="データ変更", command=change_data_button_click)
    change_data_button.pack(fill='x', padx=20, pady=20)

    #データ削除ボタンの設定
    delete_data_button = tk.Button(root, text="データ削除", command=delete_data_button_click)
    delete_data_button.pack(fill='x', padx=20, pady=20)

    #一覧ボタン
    list_button = tk.Button(root, text="一覧", command=list_click_window)
    list_button.pack(side='right', padx=20, pady=20)

    #終了ボタンの設定
    quit_button = tk.Button(root, text="終了", command=quit_button_click)
    quit_button.pack(side='bottom')

    root.mainloop()
#########################################

############データ登録ウィンドウ###########
def input_window():

    #確定ボタンのアクション(SQLにデータを追加する)
    def ok_click():
        number = 0
        item_name = str(input_item_name.get())
        price_value = int(input_price_value.get())
        update_database(item_name, price_value, number)
        root.destroy()
        root_window()

    #戻るボタンのアクション
    def back_click():
        root.destroy()
        root_window()

    #メインウィンドウの設定
    root = tk.Tk()
    root.title("データ登録画面")
    root.geometry("500x300")
    
    #商品名の入力欄の設定
    frame1 = tk.Frame(root, pady=20)
    frame1.pack()
    label_item_name = tk.Label(frame1, font=("",14), text="商品名")
    label_item_name.pack(side="left")
    input_item_name = tk.Entry(frame1, font=("",14), justify="center", width=15)
    input_item_name.pack(side="left")

    #価格の入力欄の設定
    frame2 = tk.Frame(root, pady=20)
    frame2.pack()
    label_price = tk.Label(frame2, font=("",14), text="価格")
    label_price.pack(side="left")
    input_price_value = tk.Entry(frame2, font=("",14), justify="center", width=15)
    input_price_value.pack(side="left")

    #確定ボタン
    ok_button = tk.Button(root, text="確定", command=ok_click)
    ok_button.pack(side='right', padx=20, pady=40)
    
    #戻るボタン
    back_button = tk.Button(root, text="戻る", command=back_click)
    back_button.pack(side='left', padx=20, pady=40)

    root.mainloop()
##########################################

###########データ変更ウィンドウ############
def change_window():
    
   #確定ボタンのアクション
    def ok_click():
        global name_id
        number = 1
        name_id = int(input_item_id.get())
        item_name = str(input_item_name.get())
        price_value = int(input_price_value.get())
        update_database(item_name, price_value, number)
        root.destroy()
        root_window()

    #戻るボタンのアクション
    def back_click():
        root.destroy()
        root_window()
    
    #メインウィンドウの設定
    root = tk.Tk()
    root.title("データ更新画面")
    root.geometry("500x300")

    #戻るボタン
    back_button = tk.Button(root, text="戻る", command=back_click)
    back_button.pack(side='left', padx=20, pady=40)

    #商品IDの入力欄の設定
    frame = tk.Frame(root, pady=20)
    frame.pack()
    label_item_id = tk.Label(frame, font=("",14), text="商品ID")
    label_item_id.pack(side="left")
    input_item_id = tk.Entry(frame, font=("",14), justify="center", width=15)
    input_item_id.pack(side="left")

    #商品名の入力欄の設定
    frame1 = tk.Frame(root, pady=20)
    frame1.pack()
    label_item_name = tk.Label(frame1, font=("",14), text="商品名")
    label_item_name.pack(side="left")
    input_item_name = tk.Entry(frame1, font=("",14), justify="center", width=15)
    input_item_name.pack(side="left")
    
    #価格の入力欄の設定
    frame2 = tk.Frame(root, pady=20)
    frame2.pack()
    label_price = tk.Label(frame2, font=("",14), text="価格")
    label_price.pack(side="left")
    input_price_value = tk.Entry(frame2, font=("",14), justify="center", width=15)
    input_price_value.pack(side="left")

    #確定ボタン
    ok_button = tk.Button(root, text="確定", command=ok_click)
    ok_button.pack(side='right', padx=20, pady=40)
    
    root.mainloop()
###########################################

###########データ削除ウィンドウ#############
def delete_window():

    #確定ボタンのアクション(SQLにデータを追加する)
    def ok_click():
        global name_id
        number = 2
        name_id = int(input_item_id.get())
        #item_name = str(input_item_name.get())
        #price_value = int(input_price_value.get())
        update_database("a", 100, number)
        root.destroy()
        root_window()

    #戻るボタンのアクション
    def back_click():
        root.destroy()
        root_window()
    
    #メインウィンドウの設定
    root = tk.Tk()
    root.title("データ削除画面")
    root.geometry("500x300")

    #商品IDの入力欄の設定
    frame = tk.Frame(root, pady=20)
    frame.pack()
    label_item_id = tk.Label(frame, font=("",14), text="商品ID")
    label_item_id.pack(side="left")
    input_item_id = tk.Entry(frame, font=("",14), justify="center", width=15)
    input_item_id.pack(side="left")

    #戻るボタン
    back_button = tk.Button(root, text="戻る", command=back_click)
    back_button.pack(side='left', padx=20, pady=40)

    #確定ボタン
    ok_button = tk.Button(root, text="確定", command=ok_click)
    ok_button.pack(side='right', padx=20, pady=40)

    root.mainloop()
##########################################


#########データ登録ウィンドウで確定ボタンを押したときにDB更新#########
# pt_number 0:データを追加 1:データを変更 2:データを削除
def update_database(name, value, pt_number):

    conn1 = sqlite3.connect('detail.sqlite3')
    c1 = conn1.cursor()
    conn2 = sqlite3.connect('charge.sqlite3')
    c2 = conn2.cursor()

    if value <= 5000:
            delivery_cost = 170
    elif value > 5000 and value <= 10000:
            delivery_cost = 340    
    else:
            delivery_cost = 1000
    
    commission_cost = int(value * 0.1)
    
    #データ追加
    if pt_number == 0:
        c1.execute("insert into item values(?,?)",[name, value])
        c2.execute("insert into fee values(?,?)",[delivery_cost, commission_cost])
    
    #データを変更
    elif pt_number == 1:
        c1.execute("update item set name=?, price=? where rowid=?",(name, value, name_id))
        c2.execute("update fee set delivery=?, commission=? where rowid=?",(delivery_cost, commission_cost, name_id))

    #データ削除
    elif pt_number == 2:
        c1.execute("delete from item where rowid=?", (name_id,))
        c2.execute("delete from fee where rowid=?", (name_id,))

    conn1.commit()
    conn1.close()
    conn2.commit()
    conn2.close()
        

root_window()