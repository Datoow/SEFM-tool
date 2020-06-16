def global_expl(sefm_model, minF, maxF):

  model = sefm_model
  F = model.F
  B = model.B
  print("Global explanation")
  print("w0 in model is %f." %model.w0_)
  print("Explanation of w")
  f_en = 0
  for f in range(0,F):
    print("Feature %d" %f)
    if(B[f] == 0):
      print("This feature only has one value.")
    else:
      val = minF[f]
      l = (maxF[f] - minF[f]) / B[f]
      for i in range(f_en, f_en+B[f]-1):
        print("%f" %val,"<= Feature %d <" %f,"%f: " %(val + l), model.w_[i])
        val += l
      print("%f" %val,"<= Feature %d <=" %f,"%f: " %(val + l), model.w_[f_en+B[f]-1])
    f_en += B[f]
  
  print("Explanation of w_tilde")
  f_en1 = 0
  for f1 in range(0, F):
    f_en2 = f_en1 + B[f1]
    for f2 in range(f1+1, F):
      print("Feature %d" %f1, "& Feature %d" %f2)
      if(B[f1] == 0):
        print("Feature %d only has one value." %f1)
      elif(B[f2] == 0):
        print("Feature %d only has one value." %f2)
      else:
        val1 = minF[f1]
        l1 = (maxF[f1] - minF[f1]) / B[f1]
        for i in range(f_en1, f_en1+B[f1]-1):
          val2 = minF[f2]
          l2 = (maxF[f2] - minF[f2]) / B[f2]
          for j in range(f_en2, f_en2+B[f2]-1):
            print("%f" %val1,"<= Feature %d <" %f1,"%f, " %(val1 + l1),\
            "%f" %val2,"<= Feature %d <" %f2,"%f: " %(val2 + l2), model.w_tilde[i][j])
            val2 += l2
          print("%f" %val1,"<= Feature %d <" %f1,"%f, " %(val1 + l1),\
          "%f" %val2,"<= Feature %d <=" %f2,"%f: " %(val2 + l2), model.w_tilde[i][f_en2+B[f2]-1])
          val1 += l1
        val2 = minF[f2]
        l2 = (maxF[f2] - minF[f2]) / B[f2]
        for j in range(f_en2, f_en2+B[f2]-1):
          print("%f" %val1,"<= Feature %d <=" %f1,"%f, " %(val1 + l1),\
          "%f" %val2,"<= Feature %d <" %f2,"%f: " %(val2 + l2), model.w_tilde[f_en1+B[f1]-1][j])
          val2 += l2
        print("%f" %val1,"<= Feature %d <=" %f1,"%f, " %(val1 + l1),\
        "%f" %val2,"<= Feature %d <=" %f2,"%f: " %(val2 + l2), model.w_tilde[f_en1+B[f1]-1][f_en2+B[f2]-1])
      f_en2 += B[f2]
    f_en1 += B[f1]



def local_expl(sefm_model, X, y, minF, maxF):
  model = sefm_model
  F = model.F
  B = model.B
  list_mean = []
  list_val = []
  print("Local explanation")
  print("The label is %d." %y)
  print("w0 in model is %f." %model.w0_)
  list_mean.append("w0")
  list_val.append(model.w0_)
  f_en = 0
  for f in range(0, F):
    print("Feature %d" %f)
    if(B[f] == 0):
      print("This feature only has one value.")
    else:
      l = (maxF[f] - minF[f]) / B[f]
      num = int((X[f] - minF[f])/l)
      val = minF[f] + num* l
      if num == B[f]:
        num -= 1
      if num < B[f]-1:
        print("%f" %val,"<= Feature %d <" %f,"%f: " %(val + l), model.w_[f_en+num])
        list_mean.append("%f <= Feature %d < %f" %(val, f, val + l))
        list_val.append(model.w_[f_en+num])
      else:
        print("%f" %val,"<= Feature %d <=" %f,"%f: " %(val + l), model.w_[f_en+num])
        list_mean.append("%f <= Feature %d <= %f" %(val, f, val + l))
        list_val.append(model.w_[f_en+num])
    f_en += B[f]
  
  f_en1 = 0
  for f1 in range(0, F):
    f_en2 = f_en1 + B[f1]
    for f2 in range(f1+1, F):
      print("Feature %d" %f1, "& Feature %d" %f2)
      if(B[f1] == 0):
        print("Feature %d only has one value." %f1)
      elif(B[f2] == 0):
        print("Feature %d only has one value." %f2)
      else:
        l1 = (maxF[f1] - minF[f1]) / B[f1]
        num1 = int((X[f1] - minF[f1])/l1)
        val1 = minF[f1] + num1* l1
        if num1 == B[f1]:
          num1 -= 1
        l2 = (maxF[f2] - minF[f2]) / B[f2]
        num2 = int((X[f2] - minF[f2])/l2)
        val2 = minF[f2] + num2* l2
        if num2 == B[f2]:
          num2 -= 1
        if num1 < B[f1]-1:
          if num2 < B[f2]-1:
            print("%f" %val1,"<= Feature %d <" %f1,"%f, " %(val1 + l1),\
              "%f" %val2,"<= Feature %d <" %f2,"%f: " %(val2 + l2), model.w_tilde[f_en1+num1][f_en2+num2])
            list_mean.append("%f <= Feature %d < %f, %f <= Feature %d < %f" %(val1, f1, val1 + l1, val2, f2, val2 + l2))
            list_val.append(model.w_tilde[f_en1+num1][f_en2+num2])
          else:
            print("%f" %val1,"<= Feature %d <" %f1,"%f, " %(val1 + l1),\
              "%f" %val2,"<= Feature %d <=" %f2,"%f: " %(val2 + l2), model.w_tilde[f_en1+num1][f_en2+num2])
            list_mean.append("%f <= Feature %d < %f, %f <= Feature %d <= %f" %(val1, f1, val1 + l1, val2, f2, val2 + l2))
            list_val.append(model.w_tilde[f_en1+num1][f_en2+num2])
        else:
          if num2 < B[f2]-1:
            print("%f" %val1,"<= Feature %d <=" %f1,"%f, " %(val1 + l1),\
              "%f" %val2,"<= Feature %d <" %f2,"%f: " %(val2 + l2), model.w_tilde[f_en1+num1][f_en2+num2])
            list_mean.append("%f <= Feature %d <= %f, %f <= Feature %d < %f" %(val1, f1, val1 + l1, val2, f2, val2 + l2))
            list_val.append(model.w_tilde[f_en1+num1][f_en2+num2])
          else:
            print("%f" %val1,"<= Feature %d <=" %f1,"%f, " %(val1 + l1),\
              "%f" %val2,"<= Feature %d <=" %f2,"%f: " %(val2 + l2), model.w_tilde[f_en1+num1][f_en2+num2])
            list_mean.append("%f <= Feature %d <= %f, %f <= Feature %d <= %f" %(val1, f1, val1 + l1, val2, f2, val2 + l2))
            list_val.append(model.w_tilde[f_en1+num1][f_en2+num2])
      f_en2 += B[f2]
    f_en1 += B[f1]
  
  print("The list of meaning:")
  print(list_mean)
  print("The list of value:")
  print(list_val)
    
