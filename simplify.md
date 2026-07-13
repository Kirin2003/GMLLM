  ## 配置与模型目录简化建议

  - 保留一个 configs/default.yaml 作为唯一完整配置；其他实验配置只写差异，并继续利用现有 parent 继承机制。
  - 将 DeepSeek/Llama/Qwen 的差异收敛为少量字段：llm.model_name、llm.base_url、dataset.vocab_dir、dataset.call_graph_filename、
    paths.models_dir、paths.results_dir、paths.prefix。

  - 模型输出建议统一成 models/{profile}/base_model.pt 和 models/{profile}/incremental_YYYY-MM.pt，替代 models_deepseek/、models_llama2/ 等平铺
    目录。
  - 检查一下每个配置是不是都用到了。报告没被用到的配置，但不要修改。