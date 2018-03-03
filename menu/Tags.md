---
layout: page
title: Tags
---
<ul class="posts">
  {% for post in site.posts %}

    {% unless post.next %}
      <h3>{{ post.categories }}</h3>
    {% else %}
      {% capture categories %}{{ post.categories}}{% endcapture %}
      {% capture ncategories %}{{ post.next.categories}}{% endcapture %}
      {% if categories != ncategories %}
        <h3>{{ post.categories}}</h3>
      {% endif %}
    {% endunless %}

    <li itemscope>
      <a href="{{ site.github.url }}{{ post.url }}">{{ post.title }}</a>
      <p class="post-date"><span><i class="fa fa-calendar" aria-hidden="true"></i> {{ post.date | date: "%B %-d" }} - <i class="fa fa-clock-o" aria-hidden="true"></i> {% include read-time.html %}</span></p>
    </li>

  {% endfor %}
</ul>
