-- -model /Users/fahim/Desktop/tmp/seq2seq-attn/testmodels/cpu_model.t7 -output_file tmpfile.out -src_dict  /Users/fahim/Desktop/tmp/seq2seq-attn/testmodels/pos2demo.src.dict -targ_dict /Users/fahim/Desktop/tmp/seq2seq-attn/testmodels/pos2demo.targ.dict 

local beam = require 's2sa.beam'

function string:split(sep)
   local sep, fields = sep or ":", {}
   local pattern = string.format("([^%s]+)", sep)
   self:gsub(pattern, function(c) fields[#fields+1] = c end)
   return fields
end

function main()
  beam.init(arg)
  local opt = beam.getOptions()

  assert(path.exists(opt.src_file), 'src_file does not exist')

  local file = io.open(opt.src_file, "r")
  local out_file = io.open(opt.output_file,'w')
  
  mask = {}
  for L = 1, opt.model_opt.num_layers do
    table.insert(mask, torch.ByteTensor(500):fill(0)) -- layer 1 forward
  end
  if opt.model_opt.brnn == 1 then
    for L = 1, opt.model_opt.num_layers do
      table.insert(mask, torch.ByteTensor(500):fill(0)) -- layer 1 forward
    end
  end

  local neurons_to_mask = opt.mask:split(",")
  for i = 1, #neurons_to_mask do
    print("Masking",neurons_to_mask[i])
    local n = tonumber(neurons_to_mask[i])
    local order = math.floor(n/opt.model_opt.rnn_size)+1
    n = n - (opt.model_opt.rnn_size * (order-1))
    print("Order", order, "| Neuron", n)
    mask[order][n+1] = 1
  end
  
  for line in file:lines() do
    result, nbests = beam.search_with_mask(line, mask)
    out_file:write(result .. '\n')

    for n = 1, #nbests do
      out_file:write(nbests[n] .. '\n')
    end
  end

  print(string.format("PRED AVG SCORE: %.4f, PRED PPL: %.4f", pred_score_total / pred_words_total,
    math.exp(-pred_score_total/pred_words_total)))
  if opt.score_gold == 1 then
    print(string.format("GOLD AVG SCORE: %.4f, GOLD PPL: %.4f",
      gold_score_total / gold_words_total,
      math.exp(-gold_score_total/gold_words_total)))
  end
  out_file:close()
end

main()
