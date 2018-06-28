-- Takes the same arguments as (evaluate), but instead of generating a translation
-- simply generate a table of encodings.
--
-- Run this on the same input file for multiple models in order to do a representation
-- comparison.
require 'cutorch'

require 'nn'

local beam = require 's2sa.beam'

function main()
  beam.init(arg)
  local opt = beam.getOptions()

  assert(path.exists(opt.src_file), 'src_file does not exist')
  if(opt.extract == "dec") then
    assert(path.exists(opt.targ_file))
  end

  local src_file = io.open(opt.src_file, "r")
  local targ_file
  local targ_lines
  if opt.extract == "dec" then
    targ_file = io.open(opt.targ_file, "r")
    targ_lines = targ_file:lines()
  end

  local encodings = {}
  local total_token_length = 0

  -- Encode each line in the input sample file
  local sent_idx = 1
  local skipped = 0
  for line in src_file:lines() do
    print("Processing", sent_idx)
    local encoding
    if opt.extract == "enc" then
      encoding = beam.encode(line, opt.extract_layer)
    else 
      encoding = beam.decode(line, targ_lines(), opt.extract_layer)
    end

    if encoding ~= nil then
      table.insert(encodings, nn.utils.recursiveType(encoding[1], 'torch.DoubleTensor')) -- encoding[1] should be size_l x rnn_size
      total_token_length = total_token_length + encoding:size()[2]
    else
      print('Skipping line because it is too long:')
      print(line)
      skipped = skipped + 1
    end

    sent_idx = sent_idx + 1
  end

  print("Saved", total_token_length, "token descriptions")
  print("Skipped", skipped, "lines")

  -- Save the encodings
  torch.save(opt.output_file, {
    ['encodings'] = encodings,
    ['sample_length'] = total_token_length
  })
end

main()
